#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from multiprocessing.context import assert_spawning
import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.utils.boxes import bboxes_iou
from yolox.utils.augmentations import letterbox, random_perspective

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision.transforms as transforms


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        roadgt="roadgt",
        # img_size=(416, 416),
        img_size=(640, 640),
        preproc=None,
        cache=False,
        is_train=True,
        segcls = 2
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "kittiseg")
        # print(data_dir)
        self.data_dir = data_dir
        self.json_file = json_file
        self.roadgt = roadgt
        self.mask_root = Path(os.path.join(data_dir, roadgt))
        # self.mask_list = self.mask_root.iterdir()
        self.segcls = segcls
        self.is_train = is_train
        self.Tensor = transforms.ToTensor()

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        # print(self.coco)
       
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        # print(self.coco.getCatIds())
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        # print(self.annotations)
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        # print(cache_file)
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):

        # #seg mask 
        # for mask in tqdm((list.mask_list)):
        #     mask_path
        
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        # print(im_ann["file_name"])

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        # print("anno_id" +str(anno_ids))
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            # z = obj["z"]

            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        
        res = np.zeros((num_objs, 6))


        for ix, obj in enumerate(objs):


            # print(obj["category_id"])
            cls = self.class_ids.index(obj["category_id"])
            # print(self.class_ids)
            # print(cls)
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = obj["z"]
            res[ix, 5] = cls
        # print(res)

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        # resized_info = (192,640)



        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        mask_path = self.mask_root /  (im_ann["file_name"].replace(".jpg",".png"))


        if self.segcls == 3:	
            seg_label = cv2.imread(str(mask_path))	
        else:	
            seg_label = cv2.imread(str(mask_path), 0)

        resized_shape = self.img_size	

        if isinstance(resized_shape, list):	
            resized_shape = max(resized_shape)	

        
        # h0, w0 = img.shape[:2]  # orig hw	
        # print(h0, w0)
        # print(r)
        if r != 1:  # always resize down, only resize up if training with augmentation	
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR	


        # img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)	
        seg_label = cv2.resize(seg_label, (int(width * r), int(height * r)), interpolation=interp)	
        # seg_label = cv2.resize(seg_label, (int(640), int(192)), interpolation=interp)	




    

        # print(resized_shape)
        # print(img_info, resized_info)

        # print(res[ix, 5])


        return (res, img_info, resized_info, file_name, seg_label)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        # resized_img = cv2.resize(
        #     img,
        #     (640,192),
        #     interpolation=cv2.INTER_LINEAR,
        # ).astype(np.uint8)




        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)
        # print(img_file)

        img = cv2.imread(img_file)
        assert img is not None

        return img
    def load_roadgt(self, index):

        file_name = self.annotations[index][3]


    def pull_item(self, index):

        id_ = self.ids[index]

        # print(id_)
        res, img_info, resized_info, _, seg_label= self.annotations[index]
        # print(res, img_info, resized_info, seg_label)

        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
        # print(res[0:4])
        # vis_res = vis(img, res[:, 0:4], res[:,4])
        # cv2.imwrite("test.jpg", vis_res)
        # assert 1 == 0, f"{res[:, 0:4]}"

        h0,w0 = img_info
        h,w = resized_info

        print(img.shape)
        print(h0,w0)
        print(h,w)



        (img,  seg_label), ratio, pad = letterbox((img,  seg_label),  resized_info, auto=True, scaleup=self.is_train)	
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP 

        print(img.shape)
        
        print("----------")



        # print(self.input_dim)
        # print(h,w)

        # r = min(self.input_dim[0] / h, self.input_dim[1] / w)

        # print(int(w * r), int(h * r))

        
	    # labels_out = torch.zeros((len(labels), 6))
        # if len(labels):
        #     labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        # print(res.shape)
        # print(seg_label.shape)

        # print(seg_label.shape)

        # print("-------------------------------------")
        # print(seg_label.shape)
        # print("-------------------------------------")
       

        # if self.segcls == 3:
        #     # print(seg_label[:,:,2])
        #     _,seg_label[:,:,0]= cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
        #     _,seg_label[:,:,1] = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
        #     _,seg_label[:,:,2]= cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        #     # print(seg_label[:,:,2])
        # else:
        #     _,seg_label = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)


        # # print(seg_label.shape)

        # print(seg_label.shape)


        # print(w,h)

        # seg_label = cv2.resize( 
        #     seg_label,  
        # (self.img_size[1],self.img_size[0]),
        # interpolation=cv2.INTER_NEAREST
        # ).astype(np.uint8)	


        
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

        
        seg_label = seg_label.numpy()




        # print(res.shape)
        # print(type(res))
        # print(seg_label.shape)
        # a_list = seg_label.tolist()
        # print(a_list.shape)

        # test = [res, seg_label]
        # print(test.shape)


        # print(res.shape)

        # print(seg_label)





        return img, res.copy(), img_info, np.array([id_]), seg_label

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id, seg_target = self.pull_item(index)



        # print(img, target, img_info, img_id)
        # bboxes = target[0:4]
        # print(target[0:4])
        # assert 1 == 0
        # cls = 0 
        # vis_res = vis(img, bboxes, cls,self.cls_names)



        if self.preproc is not None:
            img, target, seg_target = self.preproc(img, target, self.input_dim, seg_target )


        return img, target, img_info, img_id, seg_target

'''

#單純確認gt有沒有正確
def vis(img, boxes, z):

    for i in range(len(boxes)):
        box = boxes[i]
        # cls_id = int(cls_ids[i])s
        # score = scores[i]

        az = float(z[i])
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        text = ' {}:{:.1f}m'.format("D",az)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font,0.4, 1)[0]

        # color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        blk = np.zeros(img.shape, np.uint8)  
        cv2.rectangle(
            blk,
            (x1-1, y1),
            (x1 - int(txt_size[0]), y1 - txt_size[1] - 1),
            (0,255,0),
            -1
        )
        img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)
        cv2.putText(img, text, (x1 - txt_size[0], y1), font, 0.4, (255,255,255), thickness=1)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


'''