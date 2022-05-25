import os
import os.path as op
import re
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_path(path):
    if not op.exists(path):
        os.makedirs(path)


def pil2cv(img):
    return cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR)


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def vis_seg_contours(img, label, color):
    img = pil2cv(img)
    mask_img = np.zeros((label.shape[0], label.shape[1], 3))

    mask_img[..., 0] = (label > 0) * 255
    mask_img = pil2cv(mask_img)

    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=1)

    return np.array(cv2pil(img), np.uint8)


def draw_contour(img, mask, label_unique, vol_file, save_path):
    print(len(label_unique))
    save_path = os.path.join(save_path, vol_file)
    create_path(save_path)

    img = np.squeeze(img)
    mask = np.squeeze(mask)
    colors = [[0, 191, 255], [255, 255, 0], [173, 225, 61], [151, 0, 255], [243, 223, 48], [0, 117, 255], [58, 184, 14],
              [86, 67, 140], [255, 193, 37], [0, 255, 0], [115, 154, 81], [86, 255, 234], [255, 255, 255],
              [205, 38, 38], [255, 140, 0], [255, 236, 139], [0, 0, 255], [102, 205, 170], [127, 255, 212],
              [139, 101, 8], [139, 105, 105], [0, 0, 205], [106, 90, 205], [0, 250, 154], [173, 255, 47], [0, 255, 255],
              [0, 229, 238], [25, 25, 112]]
    x, y, z = img.shape
    for i in range(z):
        label_gen_z = mask[:, :, i]
        img_gen_z = img[:, :, i] * 255

        for index, labeli in enumerate(label_unique):
            label_l = np.zeros_like(label_gen_z)
            label_l[label_gen_z == labeli] = 1
            img_gen_z = vis_seg_contours(img_gen_z, label_l, colors[index])

        cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), img_gen_z)
