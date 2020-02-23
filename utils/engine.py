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
        except:
            print("reading failed on ", video_path)
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
    @staticmethod
    def extract_video(video_path):
        from settings import face_cache_path
        filename = video_path.split('/')[-1].split('.')[0]
        cache_path = os.path.join(face_cache_path, filename)
        if os.path.exists(cache_path):
            ret = {}
            for root, subdirs, files in os.walk(cache_path):
                for file in files:
                    face = cv2.cvtColor(np.load(os.path.join(root, file)), cv2.COLOR_BGR2RGB)
                    ret[int(file.split(".")[0])] = cv2.resize(face, 224)
            return ret
        else:
            raise Exception("cache not found")

    @staticmethod
    def get_faces(cache_path):
        faces = [cv2.cvtColor(np.load(fn), cv2.COLOR_BGR2RGB) for fn in cache_path]
        return faces


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
        return [], [], []

    @staticmethod
    def _choice(objs, options):
        ret = []
        for obj, opt in zip(objs, options):
            if opt:
                ret.append(obj)
        if len(ret) == 1:
            return ret[0]
        return ret

    def get_faces(self, images, with_person_num=False, only_one=True, with_bbox=False):
        faces, nums, bboxes = self._get(images)
        assert len(faces) == len(nums) == len(bboxes)
        if only_one:
            ret_faces = [face[0] for face in faces if len(face) > 0]
            ret_nums = [num for num, face in zip(nums, faces) if len(face) > 0]
            ret_bboxes = [box[0] for box, face in zip(bboxes, faces) if len(face) > 0]

        return self._choice([ret_faces, ret_nums, ret_bboxes], [True, with_person_num, with_bbox])

    def get_face(self, image, with_person_num=False, only_one=True, with_bbox=False):
        faces, nums, bboxes = self.get_faces(np.array([image]), with_person_num=True, only_one=False, with_bbox=True)
        faces, nums = faces[0], nums[0]
        if only_one:
            if len(faces) > 0:
                faces = faces[0]
                bboxes = bboxes[0]
            else:
                faces = None
                bboxes = None
        return self._choice([faces, nums, bboxes], [True, with_person_num, with_bbox])


class MTCNN_extractor(Face_extractor):
    def __init__(self, down_sample=2, batch_size=5, my_device=device, keep_empty=False):
        super().__init__()
        self.extractor = MTCNN(keep_all=True, device=my_device, min_face_size=80 // down_sample, ).eval()
        self.down_sample = down_sample
        self.batch_size = batch_size
        self.keep_empty = keep_empty

    def _get(self, images):
        face_list = []
        person_nums = []
        bboxes = []
        for start in range(0, len(images), self.batch_size):
            ret_face, ret_person, ret_box = self._limited_get(images[start: start + self.batch_size])
            face_list += ret_face
            person_nums += ret_person
            bboxes += ret_box
        return face_list, person_nums, bboxes

    def _limited_get(self, images):
        h, w = images.shape[1:3]
        pils = [Image.fromarray(img).resize((w // self.down_sample, h // self.down_sample)) for img in images]
        bboxes, probs = self.extractor.detect(pils)

        facelist, person_nums, box_list = [], [], []
        for boxes, img, prob in zip(bboxes, images, probs):
            if boxes is not None:
                facelist.append([self._rectang_crop(img, box) for box in boxes * self.down_sample])
                box_list.append([self._get_boundingbox(box, w, h) for box in boxes * self.down_sample])
                person_nums.append(np.sum(prob > 0.9))
            elif self.keep_empty:
                facelist.append([])
                box_list.append([])
                person_nums.append(0)

        assert len(person_nums) == len(facelist)
        return facelist, person_nums, box_list


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


class BlazeFace_extractor(Face_extractor):
    def __init__(self, blaze_weight, anchors, scale:float=1.0):
        super().__init__()
        face_detector = BlazeFace().to(device)
        face_detector.load_weights(blaze_weight)
        face_detector.load_anchors(anchors)
        _ = face_detector.train(False)
        self.extractor = FaceExtractor(face_detector, margin=scale-1)

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
        pass

    @staticmethod
    def data_transform():
        # transform to Tensor
        pre_trained_mean, pre_trained_std = [0.439, 0.328, 0.304], [0.232, 0.206, 0.201]
        return Compose([ToTensor(), Normalize(pre_trained_mean, pre_trained_std)])

    @staticmethod
    def tta(pil_img):
        assert pil_img.size == (224, 224)
        return [pil_img, RandomHorizontalFlip(p=1)(pil_img)]

    @staticmethod
    def predict(batch):
        print(batch[0])
        return 0.5

    @staticmethod
    def getx(faoa):
        l = np.float64(0)
        r = np.float64(1)
        while r - l > 5e-8:
            mid = (l + r) / 2
            if mid ** faoa > 1 - mid:
                r = mid
            else:
                l = mid
        return (r + l) / 2

    def give_predict(self, y):
        y = y.clip(5e-8, 1 - 5e-8)
        faoa = np.sum(np.log(1 - y)) / np.sum(np.log(y))
        ret = self.getx(faoa)
        if ret > 0.7 and len(y[y < 0.5]) > 0:
            return self.give_predict(y[y > 0.5])
        return ret

    def test(self, shape=(1, 3, 224, 224)):
        return self.predict(torch.rand(shape))


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
