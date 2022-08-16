#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
from cv2 import resize
import numpy as np

from yolox.utils import xyxy2cxcywh
import torchvision.transforms as transforms
from yolox.utils.augmentations import letterbox_for_img


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size


    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets

from yolox.utils import vis
import torch
def _mirror(image, boxes, segs, prob=0.5):

    # test=torch.from_numpy(segs)
    # segto1=test.max(axis=0)[1].cpu().numpy()
    # image = cv2.resize(image, (640, 192), interpolation=cv2.INTER_LINEAR)	
    # print(segto1.shape)
    # vis_res, seg_mask = vis(image, boxes, [0.4195, 0.3959, 0.3038, 0.2968, 0.2968,0], [0., 0., 0., 0., 0.,0.], 0.25, ('Car', 'Pedestrian', 'Cyclist'), segto1)
    #vis(img, bboxes, scores, cls, cls_conf, self.cls_names, seg)

    _, width, _ = image.shape
    if random.random() < prob:
        
        image = image[:, ::-1].copy()
        # segs = segs[:, ::-1].copy()


        # segs圖片轉換成np.array的處理方式
        segs = segs[:, :, ::-1].copy() if len(segs) else np.array([])

        # segs = np.fliplr(segs).copy() if len(segs) else np.array([])
        # cv2.imwrite("datasets/news.jpg",segs)
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        # seg_mask = np.fliplr(seg_mask).copy()


    return image, boxes, segs


def preproc(img, input_size, seg_target, swap=(2, 0, 1)):


    # print(img.shape)


    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114


    # print(padded_img.shape)



    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    # print(r)


    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    # print(resized_img.shape)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # print(padded_img.shape)

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    '''
    h0, w0 = img.shape[:2]
    if (h0 != 384 and w0 != 640 ) : #代表沒有被letter box轉換過
        
        # Padded resize
        # print(img.shape)
        img, ratio, pad = letterbox_for_img(img, (640,640), auto=True)
        # print(img.shape)
        h, w = img.shape[:2]
        # shapes = (h0, w0), ((h / h0, w / w0), pad)

        r = min(h / h0, w / w0)

        # print(r)
    
    else : 

        r = 1

    '''


    # Convert
    #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # padded_img = img.transpose(swap)
    # padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # print(padded_img.shape)
    # print(len(seg_target))
    # print(seg_target.shape)

    '''
    
    print(img.shape)
    if len(seg_target) > 0:
        print(seg_target.shape)	
        h, w = seg_target.shape	

        #修改segcls 記得改掉2
        padded_seg = np.zeros((int(input_size[0]), int(input_size[1])), dtype=np.uint8)	
        resized_seg = cv2.resize(	
        seg_target, (int(w * r), int(h * r)),	
        interpolation=cv2.INTER_NEAREST).astype(np.uint8)	
        print(resized_seg.shape)
        # if len(seg_target.shape) == 2:
            # print(seg_target.shape)
            # seg_target = np.expand_dims(seg_target, axis=-1)
            # print(seg_target.shape)
        padded_seg[ : int(h * r), : int(w * r)] = seg_target
        print(padded_seg.shape)
        # padded_seg = padded_seg.transpose(swap)
        padded_seg = np.ascontiguousarray(padded_seg, dtype=np.float32)	
    else:	
        padded_seg = seg_target	

    print(type(padded_img))
    print(type(padded_seg))

    '''
    # print(padded_img.shape)
    # print(padded_seg.shape)
    padded_seg = seg_target

    return padded_img, r, padded_seg


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, segcls=2):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.segcls = segcls


    def __call__(self, image, targets, input_dim, seg_targets):

        self.Tensor = transforms.ToTensor()

        # print("-----------")
        # print(seg_targets.shape)
        # print(seg_targets)
        # print("----------")

        boxes = targets[:, :4].copy()
        z = targets[:, 4].copy()
        labels = targets[:, 5].copy()
        seg = seg_targets.copy() if self.segcls > 0 else np.array([])
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)

            if self.segcls > 0:	
                h, w, _ = image.shape	
                seg_targets = seg_targets * 0	
            else:	
                seg_targets = np.array([])

            image, r_o, seg_targets = preproc(image, input_dim, seg_targets)

            '''
            #處理seg_label  將圖片轉換成np.arrary再轉換成tensor
            seg_label = seg_targets
            _,h,w = seg_label.shape
            if self.segcls == 3:
                _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
                _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
                _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
            else:
                _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
                _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)

                
            if self.segcls == 3:
                seg0 = self.Tensor(seg0)
            seg1 = self.Tensor(seg1)
            seg2 = self.Tensor(seg2)


            seg_label = np.zeros((h, w, 1))

            if self.segcls == 3:
                seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
            else:
                seg_label = torch.stack((seg2[0], seg1[0]),0)
            
            seg_targets = seg_label.numpy()

            '''



      
            return image, targets, seg_targets

        # print("-----------")
        # print(seg_targets)
        # print("----------")

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        z_o = targets_o[:, 4]
        labels_o = targets_o[:, 5]
        # print(targets_o)
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)
        segs_o = seg_targets.copy() if self.segcls > 0 else np.array([])

        



        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes, seg_t = _mirror(image, boxes, seg, self.flip_prob)


        # print("-----------")
        # print(seg_t)
        # print(seg_t.shape)
        # print("----------")

        height, width, _ = image_t.shape
        image_t, r_, seg_t= preproc(image_t, input_dim, seg_t)

        # print("-----------")
        # print(seg_t)
        # print(seg_t.shape)
        # print("----------")


        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        z_t = z[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o, segs_o = preproc(image_o, input_dim, segs_o)
            boxes_o *= r_o
            boxes_t = boxes_o
            z_t = z_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)
        z_t = np.expand_dims(z_t, 1)

        # print(boxes_t.shape)
        # print(boxes_t)
        # print(z_t.shape)
        boxes_t_z = np.hstack((boxes_t,z_t))
        # print(boxes_t_Z)

        # print(labels_t.shape)
        # print(labels_t)

        targets_t = np.hstack((labels_t, boxes_t_z))
        # print(labels_t)
        # print(targets_t)
        # print(targets_t)
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]

        # print(padded_labels)
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)


        # print( seg_t )
        # print(type(seg_t))
        # print( seg_t.shape)
        # print(padded_labels.shape)

        # print(targets)
        # print(targets_t)
        # print(padded_labels)


        # #處理seg_label  將圖片轉換成np.arrary再轉換成tensor
        # seg_label = seg_t

        
        # h,w = seg_label.shape
        # print(h,w)

        # print(seg_label)


        # if self.segcls == 3:
        #     _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
        #     _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
        #     _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # else:
        #     _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
        #     _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)

            
        # if self.segcls == 3:
        #     seg0 = self.Tensor(seg0)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)

        

        # seg_label = np.zeros((h, w, 1))

        # if self.segcls == 3:
        #     seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        # else:
        #     seg_label = torch.stack((seg2[0], seg1[0]),0)

        
        # seg_t = seg_label.numpy()

        # print(seg_t)



        return image_t, padded_labels, seg_t


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size,seg ):

        # print(img, input_size, np.array([]), self.swap)
        img, _, seg = preproc(img, input_size, seg, self.swap) 
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    

        return img, np.zeros((1, 5)), np.array([])
