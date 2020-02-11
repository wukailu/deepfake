import cv2
import torch
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from skvideo.io import vread

# 裁方形的bbox
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb / 2), 0)
    y1 = max(int(center_y - size_bb / 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(rgb_image, output_image_size: int, face_detector):
    faces = get_face_crop(face_detector, rgb_image)

    for i in range(0, len(faces)):
        faces[i] = cv2.resize(faces[i], (output_image_size, output_image_size))
    return faces


def get_face_crop(face_detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)

    height, width = image.shape[:2]

    ret_faces = []
    for face in faces:
        x, y, size = get_boundingbox(face, width, height)
        ret_faces.append(image[y:y + size, x:x + size])
    return ret_faces


def visualize_metrics(records, extra_metric, name):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    axes[0].plot(list(range(len(records.train_losses))), records.train_losses, label='train')
    axes[0].plot(list(range(len(records.train_losses_wo_dropout))), records.train_losses_wo_dropout,
                 label='train w/o dropout')
    axes[0].plot(list(range(len(records.val_losses))), records.val_losses, label='val')
    axes[0].set_title('loss')
    axes[0].legend()

    axes[1].plot(list(range(len(records.train_accs))), records.train_accs, label='train')
    axes[1].plot(list(range(len(records.train_accs_wo_dropout))), records.train_accs_wo_dropout,
                 label='train w/o dropout')
    axes[1].plot(list(range(len(records.val_accs))), records.val_accs, label='val')
    axes[1].axhline(y=0.5, color='g', ls='--')
    axes[1].axhline(y=0.667, color='r', ls='--')
    axes[1].set_title('acc')
    axes[1].legend()

    axes[2].plot(list(range(len(records.train_custom_metrics))), records.train_custom_metrics, label='train')
    axes[2].plot(list(range(len(records.train_custom_metrics_wo_dropout))), records.train_custom_metrics_wo_dropout,
                 label='train w/o dropout')
    axes[2].plot(list(range(len(records.val_custom_metrics))), records.val_custom_metrics, label='val')
    axes[2].axhline(y=0.5, color='g', ls='--')
    axes[2].axhline(y=0.5, color='r', ls='--')
    axes[2].set_title(f'{extra_metric.__name__}')
    axes[2].legend()

    axes[3].plot(list(range(len(records.lrs))), records.lrs)
    _ = axes[3].set_title('lr')
    plt.tight_layout()
    plt.savefig(name, format='png')


def display_predictions_on_image(model, precomputed_cached_path, val_iter, name):
    # val
    model.eval()
    data = next(val_iter)

    inputs, labels = get_input_with_label(data)
    img_files = data['real_file'] + data['fake_file']

    with torch.no_grad():
        outputs = model(inputs)
        outputs_predicbilty = outputs  # torch.nn.functional.softmax(outputs, dim=1)
        assert len(outputs_predicbilty) == len(outputs), f'proba shape: {len(outputs_predicbilty)}'

        _, predicted = torch.max(outputs.data, 1)

    numbers = min(labels.size(0), 100)
    nrows = int(numbers ** 0.5)
    ncols = int(np.ceil(numbers / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 40))
    step = 0
    for i in range(nrows):
        for j in range(ncols):
            face_crop = np.load(img_files[step])
            axes[i, j].set_title(
                f'{outputs_predicbilty[step][0]:.2f},{outputs_predicbilty[step][1]:.2f}|{predicted[step]}|{labels[step]}')
            axes[i, j].imshow(face_crop)
            step += 1
            if step == numbers:
                break
    plt.title('predicted probability real, fake | prediction | label (0: real 1: fake)')
    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close(fig)


def get_input_with_label(data: dict):
    batch_size = data['real'].shape[0]
    inputs = torch.cat((data['real'], data['fake'])).cuda()
    labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).long().cuda()
    return inputs, labels


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    make_square_image(resized)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    top = 0
    bottom = size - h
    left = 0
    right = size - w
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


# def get_video(video_file):
#     frames = vread(video_file)
#
#     #     cap = cv.VideoCapture(video_file)
#     #     frames = []
#     #     while(cap.isOpened()):
#     #         ret, frame = cap.read()
#     #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #         if ret==True:
#     #             frames.append(frame)
#     #             if cv2.waitKey(1) & 0xFF == ord('q'):
#     #                 break
#     #         else:
#     #             break
#     #     cap.release()
#
#     return frames