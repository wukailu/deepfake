import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import sys

sys.path.append("pkgs")
sys.path.append("utils")
from read_video import VideoReader
from face_extract import FaceExtractor
from blazeface import BlazeFace
import os
import torch
from torchvision.transforms import Normalize, RandomHorizontalFlip, ToTensor, Compose
import matplotlib.pyplot as plt
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Video_reader:
    @staticmethod
    def extract_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        assert len(frames) != 0
        return np.array(frames)

    @staticmethod
    def extract_one_frame(video_path, frame_index):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 设置要获取的帧号
        _, frame = cap.read()
        cap.release()
        if _:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            return None


class FullVideoReader:

    @staticmethod
    def extract_video(video_path):
        from skvideo.io import vread
        try:
            ret = vread(video_path)
        except Exception as e:
            print("reading failed on ", video_path, e)
            ret = np.array([])
        return ret

    def extract_one_frame(self, video_path, frame_index):
        return self.extract_video(video_path)[frame_index]


class VideoDataset(Dataset):
    def __init__(self, video_paths, sample_rate, video_reader):
        self.paths = video_paths
        self.sample_rate = sample_rate
        self.video_reader = video_reader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        frames = self.video_reader.extract_video(self.paths[idx])
        if len(frames) == 0:
            return np.array([])
        samples = np.linspace(0, len(frames) - 1, self.sample_rate).round().astype(int)
        return frames[samples]


class FastDataset(Dataset):
    def __init__(self, video_paths, sample_rate, verbose=False):
        self.paths = video_paths
        self.sample_rate = sample_rate
        self.reader = VideoReader(verbose=verbose)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        ret = self.reader.read_frames(self.paths[idx], self.sample_rate)
        if ret is None:
            return None
        my_frames, my_idxs = ret
        return my_frames


class Cache_loader:
    def __init__(self, video_paths):
        self.paths = video_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        ret = self.extract_video(self.paths[idx])
        if len(ret) == 0:
            return []
        return list(ret.values())

    @staticmethod
    def extract_video(video_path):
        from settings import face_cache_path
        filename = video_path.split('/')[-1].split('.')[0]
        cache_path = os.path.join(face_cache_path, filename)
        if os.path.exists(cache_path):
            ret = {}
            for root, subdirs, files in os.walk(cache_path):
                for file in files:
                    with open(os.path.join(root, file), 'rb') as f:
                        face = Image.open(f)
                        face.load()
                    ret[int(file.split(".")[0])] = np.array(face)
            return ret
        else:
            print("cache not found")
            return {}

    @staticmethod
    def get_faces(cache_path):
        faces = [cv2.cvtColor(np.load(fn), cv2.COLOR_BGR2RGB) for fn in cache_path]
        return faces


FaceInfo = namedtuple("FaceInfo", ['face', 'box', 'prob'])


class Face_extractor:
    def __init__(self):
        pass

    @staticmethod
    def _get_boundingbox(bbox, width, height, scale=1.2, minsize=None):
        x1, y1, x2, y2 = bbox[:4]
        if not 0.33 < (x2 - x1) / (y2 - y1) < 3:
            return np.array([0, 0, 0, 0])
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize:
            if size_bb < minsize:
                size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        x1 = max(int(center_x - size_bb / 2), 0)
        y1 = max(int(center_y - size_bb / 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return np.array([x1, y1, x1 + size_bb, y1 + size_bb]).astype(int)

    def _rectang_crop(self, image, bbox, with_bbox=False):
        height, width = image.shape[:2]
        l, t, r, b = self._get_boundingbox(bbox, width, height)
        if with_bbox:
            return image[t:b, l:r], np.array([l, t, r, b])
        return image[t:b, l:r]

    def _get(self, images):
        return []

    def get_faces(self, images, only_one=True):
        faceInfoList = self._get(images)
        if only_one:
            faceInfoList = [multiFace[0] for multiFace in faceInfoList if len(multiFace) > 0]
        return faceInfoList

    def get_face(self, image, only_one=True):
        faceInfo = self.get_faces(np.array([image]), only_one=False)[0]
        if only_one:
            if len(faceInfo) > 0:
                faceInfo = faceInfo[0]
            else:
                faceInfo = None
        return faceInfo


class MTCNN_extractor(Face_extractor):
    def __init__(self, down_sample=2, batch_size=30, my_device=device, keep_empty=False, factor=0.709, size_range=None, prob_limit=None):
        super().__init__()
        self.prob_limit = prob_limit
        self.size_range = size_range
        self.extractor = MTCNN(keep_all=True, device=my_device, min_face_size=80 // down_sample, factor=factor).eval()
        self.down_sample = down_sample
        self.batch_size = batch_size
        self.keep_empty = keep_empty

    def _get(self, images):
        ret = []
        for start in range(0, len(images), self.batch_size):
            ret.extend(self._limited_get(images[start: start + self.batch_size]))
        if self.size_range is not None:
            ret = self._filter(ret)
        return ret

    def _limited_get(self, images):
        h, w = images.shape[1:3]
        if h * w < 1280 * 720:
            down_sample = max(1, self.down_sample // 2)
        elif h * w >= 1280 * 720 * 4:
            down_sample = self.down_sample * 2
        else:
            down_sample = self.down_sample
        pils = [Image.fromarray(img).resize((w // down_sample, h // down_sample)) for img in images]
        bboxes, probs = self.extractor.detect(pils)

        ret = []
        for boxes, img, prob in zip(bboxes, images, probs):
            faceInfo = []
            if boxes is not None:
                faceInfo = [FaceInfo(face=self._rectang_crop(img, box), box=self._get_boundingbox(box, w, h), prob=p)
                            for box, p in zip(boxes * down_sample, prob)]
            elif self.keep_empty:
                continue
            ret.append(faceInfo)
        return ret

    def _filter(self, ret):
        new_ret = []
        for frames in ret:
            ret_frame = []
            for face in frames:
                size = (face.box[2]-face.box[0])*(face.box[3]-face.box[1])
                if self.size_range[0] < size < self.size_range[1] and face.prob>self.prob_limit:
                    ret_frame.append(face)
            new_ret.append(ret_frame)
        return new_ret


# TODO: change return type
class Dlib_extractor(Face_extractor):
    def __init__(self):
        import dlib
        super().__init__()
        self.extractor = dlib.get_frontal_face_detector()

    def _get(self, images):
        rets = [self.dlib_get_one_face(image) for image in images]
        person_nums = [p for f, p, box in rets]
        faces = [f for f, p, box in rets]
        bboxes = [box for f, p, box in rets]
        return faces, person_nums, bboxes

    def dlib_get_one_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.extractor(gray, 0)
        bboxes = [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]
        facelist = [self._rectang_crop(image, box) for box in bboxes]
        return facelist, len(facelist), bboxes


# TODO: change return type
class BlazeFace_extractor(Face_extractor):
    def __init__(self, blaze_weight, anchors, scale: float = 1.0):
        super().__init__()
        face_detector = BlazeFace().to(device)
        face_detector.load_weights(blaze_weight)
        face_detector.load_anchors(anchors)
        _ = face_detector.train(False)
        self.extractor = FaceExtractor(face_detector, margin=scale - 1)

    def _get(self, images):
        ret = self.extractor.process_video(images)
        person_nums = [np.sum(np.array(dic['scores']) > 0.8) for dic in ret]
        faces = [dic["faces"] for dic in ret]
        box = [dic["boxes"] for dic in ret]
        for dic in ret:
            assert len(dic["boxes"]) == len(dic["faces"])
        return faces, person_nums, box


class Inference_model:
    def __init__(self):
        self.pre_mean, self.pre_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = Compose([ToTensor(), Normalize(self.pre_mean, self.pre_std)])
        pass

    def solve(self, frames: list) -> float:
        return 0.5

    def predict_batch(self, batch) -> np.ndarray:
        return np.array([0.5]*len(batch))

    @staticmethod
    def _getx(faoa):
        l = np.float64(0)
        r = np.float64(1)
        while r - l > 5e-8:
            mid = (l + r) / 2
            if mid ** faoa > 1 - mid:
                r = mid
            else:
                l = mid
        return (r + l) / 2

    def _ensemble(self, y: list) -> float:
        y = np.array(y)
        y = y.clip(5e-8, 1 - 5e-8)
        faoa = np.sum(np.log(1 - y)) / np.sum(np.log(y))
        ret = self._getx(faoa)
        return ret

    def test(self, shape=(10, 1920, 1080, 3)):
        return self.solve(np.rand(shape))


class FaceInferenceModel(Inference_model):
    def __init__(self, face_extractor=None):
        super().__init__()
        self.face_extractor = face_extractor
        self.batch_size = 32

    def _tta(self, pil_img):
        assert pil_img.size == (224, 224)
        return [pil_img, RandomHorizontalFlip(p=1)(pil_img)]

    def solve(self, frames: list) -> float:
        assert self.face_extractor is not None
        faces = self.face_extractor.get_faces(frames, only_one=False)
        return self.solve_faces(faces)

    def solve_faces(self, faces: list) -> float:
        inputs = []
        image_id = []
        cnt = 0
        for faceInFrame in faces:
            ids = []
            for face in faceInFrame:
                ttas = []
                pils = self._tta(Image.fromarray(face.face).resize((224, 224)))
                for pic in pils:
                    inputs.append(self.transform(pic))
                    ttas.append(cnt)
                    cnt += 1
                ids.append(ttas)
            if len(ids) > 0:
                image_id.append(ids)

        output = self.predict_all(torch.stack(inputs))
        result = self._ensemble([np.max([self._ensemble([output[tta] for tta in face]) for face in frame]) for frame in image_id])
        return result

    def predict_all(self, batch) -> np.ndarray:
        ret = []
        for start in range(0, len(batch), self.batch_size):
            ret.extend(self.predict_batch(batch[start:start+self.batch_size]))
        return np.array(ret)


def show(images):
    rows = int(np.sqrt(len(images)))
    col = int(np.ceil(len(images) / rows))
    fig, axes = plt.subplots(rows, col)
    ax = np.array(axes).reshape(-1)
    for i, img in enumerate(images):
        if img is not None:
            ax[i].imshow(img)
    plt.grid(False)
    plt.show()
