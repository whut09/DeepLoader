#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math

import cv2
import numpy as np
from deeploader.util.alignment import *

KEY_ESC = 27
KEY_SPACE = 32
KEY_CR = 13

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (128, 128, 128)


def randColor():
    return [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]


def cvRectangle(img, pt1, pt2, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)


def cvRectangleR(img, box, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                  color, thickness)


def cvBox(img, box, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                  color, thickness)


def cvBBox(img, patch_points, color=COLOR_GREEN, thickness=1):
    cv2.line(img, (patch_points[0][0], patch_points[0][1]),
             (patch_points[1][0], patch_points[1][1]), color,
             thickness)
    cv2.line(img, (patch_points[1][0], patch_points[1][1]),
             (patch_points[2][0], patch_points[2][1]), color,
             thickness)
    cv2.line(img, (patch_points[2][0], patch_points[2][1]),
             (patch_points[3][0], patch_points[3][1]), color,
             thickness)
    cv2.line(img, (patch_points[3][0], patch_points[3][1]),
             (patch_points[0][0], patch_points[0][1]), color,
             thickness)


def cvSet(img, val):
    img[::] = val
    return img


def cvSetROI(dst, rect, val):
    dst[rect[0]:(rect[0] + rect[2]), rect[1]:(rect[1] + rect[3])] = val
    return dst


def cvZero(img):
    img[::] = 0
    return img


def cvCopy(src, dst, rect):
    dst[rect[0]:(rect[0] + rect[2]), rect[1]:(rect[1] + rect[3]), :] = src
    return dst


def cvCrop(src, rect):
    dst = src[int(rect[0]):int(rect[0] + rect[2]), int(rect[1]):int(rect[1] + rect[3])].copy()
    return dst


def img_fit_center(src, dst):
    rx = float(src.shape[1]) / dst.shape[1]
    ry = float(src.shape[0]) / dst.shape[0]
    # try fit x
    dx = dst.shape[1]
    dy = src.shape[0] / rx

    if dy > dst.shape[0]:
        dx = src.shape[1] / ry
        dy = dst.shape[0]
    dx = int(dx)
    dy = int(dy)
    _img = cv2.resize(src, (dx, dy))
    # print('resize to:{}'.format(_img.shape))
    ox = (dst.shape[1] - dx) // 2
    oy = (dst.shape[0] - dy) // 2
    cvCopy(_img, dst, (oy, ox, dy, dx))
    return dst


def size_fit_center(src, dst_shape):
    rx = float(src.shape[1]) / dst_shape[1]
    ry = float(src.shape[0]) / dst_shape[0]
    # try fit x
    dx = dst_shape[1]
    dy = src.shape[0] / rx
    if dy > dst_shape[0]:
        dx = src.shape[1] / ry
        dy = dst_shape[0]
    dx = int(dx)
    dy = int(dy)
    ox = (dst_shape[1] - dx) // 2
    oy = (dst_shape[0] - dy) // 2
    return [oy, ox, dy, dx]


def resize_max(src, dst):
    rx = float(src.shape[1]) / dst.shape[1]
    ry = float(src.shape[0]) / dst.shape[0]
    # try fit x
    dx = dst.shape[1]
    dy = src.shape[0] / rx
    if dy > dst.shape[0]:
        dx = src.shape[1] / ry
        dy = dst.shape[0]
    dx = int(dx)
    dy = int(dy)
    _img = cv2.resize(src, (dx, dy))
    return _img


def random_crop(src, data_shape, center=1):
    # numpy: hwc
    h, w, c = src.shape
    # data_shape chw
    if h != 256 or w != 256:
        src = cv2.resize(src, (256, 256))
        h, w, c = src.shape
    # crop
    if center:
        ox = (w - data_shape[2]) // 2
        oy = (h - data_shape[1]) // 2
    else:
        ox = np.random.randint(0, w - data_shape[2])
        oy = np.random.randint(0, h - data_shape[1])
    patch = cvCrop(src, (oy, ox, data_shape[1], data_shape[2]))
    return patch.copy()


def make_rect_mask(w, h, border=10, elastic=False, kernel='linear'):
    img = np.zeros((h, w, 1), dtype=np.float)
    h, w, c = img.shape
    for i in range(border):
        # gradient border
        ratio = float(i + 1) / border
        thick = 1
        # linear kernel
        if kernel == 'none':
            alpha = 1
        elif kernel == 'linear':
            alpha = ratio
        elif kernel.find('quad') == 0:
            alpha = math.pow(ratio, 2)
        elif kernel == 'cubic':
            alpha = math.pow(ratio, 3)
        elif kernel.find('cos') == 0:
            alpha = (1 - math.cos(ratio * math.pi)) * 0.5
        else:
            alpha = ratio
        # draw inner block
        if i == border - 1:
            thick = -1
            alpha = 1
        alpha = max(1e-5, alpha)
        cv2.rectangle(img, (i, i), (w - i, h - i), (alpha, alpha, alpha), thick)
    if elastic:
        import imgaug.augmenters as iaa
        img = iaa.ElasticTransformation(alpha=border, sigma=5).augment_image(img)
    np.clip(img, 0.0, 1.0)
    return img


def paste_image2(src, dst, dst_points, border=20, trans_type='similarity',
                 elastic=True, border_kernel='cubic'):
    """
        paste src image to dst image with border specified by dst_points
    Parameters:
    ----------
        src numpy array
            input image
        dst dst image
        dst_points: list of dst points
        trans_type: similarity OR affine, default similarity
    Return:
    -------
        dst image
    """
    # average positions of face points
    # tranform
    t = compute_similarity_transform
    if trans_type == 'affine':
        t = compute_affine_transform
    patch_h, patch_w, _ = src.shape
    patch_points = []
    patch_points.append([0, 0])
    patch_points.append([patch_w - 1, 0])
    patch_points.append([patch_w - 1, patch_h - 1])
    patch_points.append([0, patch_h - 1])
    # 0. generate mask
    mask = make_rect_mask(patch_w, patch_h, border, elastic=elastic, kernel=border_kernel)
    # 1. crop dst image
    d2s = t(dst_points, patch_points)
    dst_patch = cv2.warpAffine(dst, d2s, (patch_w, patch_h), borderMode=cv2.BORDER_REFLECT)
    # 2. blend
    blend = mask * src + (1.0 - mask) * dst_patch
    np.clip(blend, 0, 255.0)
    blend = blend.astype(np.uint8)
    # cv2.imshow('src', src)
    # cv2.imshow('d_patch', dst_patch)
    cv2.imshow('blend', blend)
    s2d = t(patch_points, dst_points)
    # cvZero(dst_patch)
    cv2.warpAffine(dst_patch, s2d, (dst.shape[1], dst.shape[0]),
                   dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def paste_image(src, dst, dst_points, border=20, trans_type='similarity',
                elastic=True, border_kernel='cubic'):
    """
        paste src image to dst image with border specified by dst_points
    Parameters:
    ----------
        src numpy array
            input image
        dst dst image
        dst_points: list of dst points
        trans_type: similarity OR affine, default similarity
    Return:
    -------
        dst image
    """
    # average positions of face points
    # tranform
    t = compute_similarity_transform
    if trans_type == 'affine':
        t = compute_affine_transform
    patch_h, patch_w, _ = src.shape
    patch_points = []
    patch_points.append([0, 0])
    patch_points.append([patch_w - 1, 0])
    patch_points.append([patch_w - 1, patch_h - 1])
    patch_points.append([0, patch_h - 1])
    # 0. generate mask
    mask = make_rect_mask(patch_w, patch_h, border, elastic=elastic, kernel=border_kernel)

    s2d = t(patch_points, dst_points)
    # cvZero(dst_patch)
    dmask = np.zeros((dst.shape[0], dst.shape[1], 1), dtype=np.float)
    cv2.warpAffine(mask, s2d, (dst.shape[1], dst.shape[0]), dst=dmask)
    dpatch = cv2.warpAffine(src, s2d, (dst.shape[1], dst.shape[0]))
    # 2. blend
    blend = dmask * dpatch + (1.0 - dmask) * dst
    np.clip(blend, 0, 255.0)
    blend = blend.astype(np.uint8)
    # cv2.imshow('src', src)
    # cv2.imshow('d_patch', dst_patch)
    # cv2.imshow('blend', blend)
    return blend
