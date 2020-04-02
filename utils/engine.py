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
from torchvision.transforms import Normalize, RandomHorizontalFlip, ToTensor, Compose, Resize
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
    def __init__(self, video_paths, sample_rate, video_reader, new_length=1, with_id=False):
        self.paths = video_paths
        self.sample_rate = sample_rate
        self.video_reader = video_reader
        self.new_length = new_length
        self.with_id = with_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        frames = self.video_reader.extract_video(self.paths[idx])
        if len(frames) == 0 and not self.with_id:
            return np.array([])
        elif len(frames) == 0 and self.with_id:
            return np.array([]), []
        samples = np.linspace(0, len(frames) - self.new_length, self.sample_rate).round().astype(int)
        samples = np.array(sorted(np.concatenate([samples + i for i in range(self.new_length)])))
        if self.with_id:
            return frames[samples], samples
        else:
            return frames[samples]


class FastDataset(Dataset):
    def __init__(self, video_paths, sample_rate, verbose=False, new_length=1):
        self.paths = video_paths
        self.sample_rate = sample_rate
        self.reader = VideoReader(verbose=verbose)
        self.new_length = new_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        ret = self.reader.read_frames(self.paths[idx], self.sample_rate, new_length=self.new_length)
        if ret is None:
            return None, None
        my_frames, my_idxs = ret
        return my_frames, np.array(my_idxs)


class Cache_loader:
    def __init__(self, video_paths, cache_folder="", path_suffix=""):
        from settings import face_cache_path
        self.paths = video_paths
        self.path_suffix = path_suffix
        if cache_folder == "":
            self.cache_folder = face_cache_path
        else:
            self.cache_folder = cache_folder

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        ret = self.extract_video(self.paths[idx])
        if len(ret) == 0:
            return []
        return list(ret.values())

    def extract_video(self, video_path):
        filename = video_path.split('/')[-1].split('.')[0]
        cache_path = os.path.join(self.cache_folder, filename, self.path_suffix)
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


FaceInfo = namedtuple("FaceInfo", ['face', 'box', 'prob', 'frame'])


class Face_extractor:
    def __init__(self):
        pass

    @staticmethod
    def _get_boundingbox(bbox, width, height, scale=1.2, min_size=None, max_size=None):
        x1, y1, x2, y2 = bbox[:4]
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if min_size and size_bb < min_size:
            size_bb = min_size
        if max_size and size_bb > max_size:
            size_bb = max_size
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        x1 = max(int(center_x - size_bb / 2), 0)
        y1 = max(int(center_y - size_bb / 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return np.array([x1, y1, x1 + size_bb, y1 + size_bb]).astype(int)

    def _rectang_crop(self, image, bbox, scale=1.2, min_size=None, max_size=None):
        height, width = image.shape[:2]
        l, t, r, b = self._get_boundingbox(bbox, width, height, scale, min_size, max_size)
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
    def __init__(self, down_sample=2, batch_size=30, my_device=device, keep_empty=False, factor=0.709, size_range=None,
                 prob_limit=None, same_bbox_size=False, scale=1.2):
        super().__init__()
        self.prob_limit = prob_limit
        self.size_range = size_range
        self.extractor = MTCNN(keep_all=True, device=my_device, min_face_size=80 // down_sample, factor=factor).eval()
        self.down_sample = down_sample
        self.batch_size = batch_size
        self.keep_empty = keep_empty
        self.same_bbox_size = same_bbox_size
        self.scale = scale

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

        clean_bboxes, clean_probs = [], []
        for boxes, prob in zip(bboxes, probs):
            if boxes is not None:
                rets = sorted([(p, box) for box, p in zip(boxes, prob)], key=lambda x: x[0])
                if len(rets) >= 2:
                    rets = rets[-1:] if rets[-1][0] - rets[-2][0] > 0.05 else rets[-2:]
                clean_bboxes.append(np.array([box for p, box in rets]))
                clean_probs.append(np.array([p for p, box in rets]))
            else:
                clean_bboxes.append([])
                clean_probs.append([])

        bsize = sorted([max(box[2] - box[0], box[3] - box[1]) * self.scale for boxes in clean_bboxes for box in
                        boxes * down_sample])
        if len(bsize) > 0:
            bsize = int(bsize[-len(bsize) // 4])  # -1//4 = -1

        ret = []
        for boxes, img, prob, idx in zip(clean_bboxes, images, clean_probs, range(len(clean_probs))):
            faceInfo = []
            if boxes is not None and len(boxes) > 0:
                min_size = bsize if self.same_bbox_size else None
                max_size = bsize if self.same_bbox_size else None

                faceInfo = [FaceInfo(face=self._rectang_crop(img, box, self.scale, min_size, max_size),
                                     box=self._get_boundingbox(box, w, h, self.scale, min_size, max_size),
                                     prob=p,
                                     frame=idx)
                            for box, p in zip(boxes * down_sample, prob)]
                faceInfo = sorted(faceInfo, key=lambda x: -x.prob)
            elif not self.keep_empty:
                continue
            ret.append(faceInfo)
        return ret

    def _filter(self, ret):
        new_ret = []
        for frames in ret:
            ret_frame = []
            for face in frames:
                size = (face.box[2] - face.box[0]) * (face.box[3] - face.box[1])
                if self.size_range[0] < size < self.size_range[1] and face.prob > self.prob_limit:
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
    def __init__(self, batch_size=32, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.pre_mean, self.pre_std = mean, std
        self.transform = Compose([Resize(224), ToTensor(), Normalize(self.pre_mean, self.pre_std)])  # , CenterErase()
        self.batch_size = 32
        pass

    def solve(self, frames: list) -> float:
        return 0.5

    def predict_batch(self, batch) -> np.ndarray:
        return np.array([0.5] * len(batch))

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

    def predict_all(self, batch) -> np.ndarray:
        ret = []
        for start in range(0, len(batch), self.batch_size):
            ret.extend(self.predict_batch(batch[start:start + self.batch_size]))
        return np.array(ret)


class CenterErase(object):
    def __init__(self, size=100):
        self.padding = size // 2

    def __call__(self, img):
        import torchvision.transforms.functional as F
        img_c, img_h, img_w = img.shape
        return F.erase(img, img_h // 2 - self.padding, img_w // 2 - self.padding, img_h // 2 + self.padding,
                       img_w // 2 + self.padding, 0, False)


class FaceInferenceModel(Inference_model):
    def __init__(self, face_extractor=None, batch_size=32):
        super().__init__()
        self.face_extractor = face_extractor
        self.batch_size = batch_size

    def _tta(self, pil_img):
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
                pils = self._tta(Image.fromarray(face.face))
                for pic in pils:
                    inputs.append(self.transform(pic))
                    ttas.append(cnt)
                    cnt += 1
                ids.append(ttas)
            if len(ids) > 0:
                image_id.append(ids)

        if len(inputs) == 0:
            return 0.5
        output = self.predict_all(torch.stack(inputs))
        result = self._ensemble(
            [np.max([self._ensemble([output[tta] for tta in face]) for face in frame]) for frame in image_id])
        return result

    def test_model(self, shape=(1, 3, 224, 224)):
        self.predict_batch(torch.zeros(shape))


def inter(box1, box2):
    x_inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    y_inter = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    return x_inter * y_inter


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def tracking_face(face_info_lists):
    tracks = []
    for face_infos in face_info_lists:
        for face in face_infos:
            best = -1
            largest_inter = 0
            for idx, track in enumerate(tracks):
                if inter(track[-1].box, face.box) / area(track[-1].box) > largest_inter:
                    best = idx
                    largest_inter = inter(track[-1].box, face.box) / area(track[-1].box)
            if best == -1:
                tracks.append([face])
            else:
                tracks[best].append(face)

    if len(tracks) == 0:
        return []
    max_len = max([len(track) for track in tracks])
    ret = []
    for track in tracks:
        if len(track) > max_len * 0.3:
            ret.append(track)
    return sorted(ret, key=lambda x: -len(x))


class MultiFrameModel(Inference_model):
    def __init__(self, batch_size=4, num_frames=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames

    def _tta(self, pil_img):
        return [pil_img]

    def _solve_pack(self, tensor_imgs):
        return torch.stack(tensor_imgs)

    def _prepare_pack(self, track):
        faces = sorted(track, key=lambda x: x.frame)
        pack_num = len(track) // self.num_frames
        ret = [[] for i in range(pack_num)]
        cnt = 0
        for i in faces:
            if len(ret[cnt]) < self.num_frames:
                ret[cnt].append(i)
            cnt = (cnt + 1) % pack_num
        return ret

    def solve_faces(self, faces: list) -> float:
        tracks = tracking_face(faces)
        pieces = [self._prepare_pack(track) for track in tracks]

        inputs = []
        image_id = []
        cnt = 0
        for one_person in pieces:
            ids = []
            for face_pack in one_person:
                pil_packs = [self._tta(Image.fromarray(imgs.face)) for imgs in face_pack]
                ttas = []
                for i in range(len(pil_packs[0])):
                    pack = [ttad[i] for ttad in pil_packs]
                    inputs.append(self._solve_pack([self.transform(img) for img in pack]))
                    ttas.append(cnt)
                    cnt += 1
                if len(ttas) > 0:
                    ids.append(ttas)
            if len(ids) > 0:
                image_id.append(ids)

        if len(inputs) == 0 or cnt == 0:
            return 0.5

        output = self.predict_all(torch.stack(inputs))
        result = np.max(
            [self._ensemble([self._ensemble([output[i] for i in ttas_]) for ttas_ in ids_]) for ids_ in image_id])
        return result

    def test_model(self, shape=(10, 1920, 1080, 3)):
        shape = (self.batch_size, self.num_frames, 3, 224, 224)
        self.predict_batch(torch.zeros(shape))


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
