#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

import sys
sys.path.append("/home/lab602.10977014/.pipeline/10977014/YOLOXt/")

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import math
import numpy as np
import shutil


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "-demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./datasets/bddyolo/val", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=None,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/custom/yolox_x.py",
        type=str,
        help="pls input your experiment description file",
    )
    # YOLOX_outputs/yolox_s_300epoch/latest_ckpt.pth
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_x/latest_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.02, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    only_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
                only_names.append(filename.split(".")[0])
    return image_names, only_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None


        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))


        return outputs, img_info
        

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio


        cls = output[:, 7]

        # print("-------------------")
        # print(cls)
        # print("-------------------")
        z = output[:,4]
        scores = output[:, 5] * output[:, 6]


        list_boxes = bboxes.tolist()
        list_z = z.tolist()
        i = 0 
        for i in range(len(list_boxes)) :
            list_boxes[i].append(list_z[i])


        
            
        vis_res = vis(img, bboxes, scores, z, cls, cls_conf, self.cls_names)

        return vis_res, list_boxes



def plot_box_z(boxes, img, color=None , line_thickness=None):


    height, width, _ = img.shape 
    

    for i in range(len(boxes)) :
        box = boxes[i]
        
        
        x_center = float(box[1]) * width
        y_center = float(box[2]) * height
        w = float(box[3]) * width 
        h = float(box[4]) * height


        x0 = round(x_center-w/2)
        y0 = round(y_center-h/2)
        x1 = round(x_center+w/2)
        y1 = round(y_center+h/2)
        az = float(box[5])
        text = ' {}:{:.1f}m'.format("D",az)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font,0.4, 1)[0]

        
        tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thicknesss
        # color = color or [random.randint(0, 255) for _ in range(3)]
        color = (0,128,0)
        c1, c2 = (x0,y0), (x1,y1)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        blk = np.zeros(img.shape, np.uint8)  
        cv2.rectangle(
            blk,
            (x1-1, y1),
            (x1 - int(txt_size[0]), y1 - txt_size[1] - 1),
            (0,255,0),
            -1
        )
        img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)

        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.putText(img, text, (x1 - txt_size[0], y1), font, 0.4, (255,255,255), thickness=1)

    # print("the num of gt :" + str(len(boxes)))
    # cv2.imwrite("./datasets/add_z.jpg", img)



def plot_box(boxes, img, color=None , line_thickness=None):


    height, width, _ = img.shape 
    

    for i in range(len(boxes)) :
        box = boxes[i]
        
        
        x_center = float(box[1]) * width
        y_center = float(box[2]) * height
        w = float(box[3]) * width 
        h = float(box[4]) * height

        x0 = round(x_center-w/2)
        y0 = round(y_center-h/2)
        x1 = round(x_center+w/2)
        y1 = round(y_center+h/2)

        
        tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thicknesss
        # color = color or [random.randint(0, 255) for _ in range(3)]
        color = (0,128,0)
        c1, c2 = (x0,y0), (x1,y1)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # print("the num of gt :" + str(len(boxes)))

def plot_box2(boxes, img, color=None , line_thickness=None):

 
    height, width, _ = img.shape 
    

    for i in range(len(boxes)) :
        box = boxes[i]
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]
        
        tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thicknesss
        # color = color or [random.randint(0, 255) for _ in range(3)]
        color = (255,0,0)
        c1, c2 = (int(x0),int(y0)), (int(x1),int(y1))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # print("the num of pred :" + str(len(boxes)))

def dist(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )


def transfer_gt( img, boxes ) :


    height, width, _ = img.shape 

    resized_boxes = []
    

    for i in range(len(boxes)) :
        box = boxes[i]
        
        
        x_center = float(box[1]) * width
        y_center = float(box[2]) * height
        w = float(box[3]) * width 
        h = float(box[4]) * height

        x0 = round(x_center-w/2)
        y0 = round(y_center-h/2)
        x1 = round(x_center+w/2)
        y1 = round(y_center+h/2)

        resized_boxes.append([int(x0), int(y0), int(x1), int(y1), False])


    return resized_boxes



def add_z(img, gt_labels, predict, match_num, current, total, image_name, name) :

    
    # ratio = img_info["ratio"]
    # img = img_info["raw_img"]

    miss_num = len(gt_labels) - len(predict)

    # plot_box(gt_labels, img )

    # plot_box2(predict, img )

    # cv2.imwrite("./datasets/merge.jpg", img)

    gt_list = transfer_gt(img,gt_labels)
    # print("============================")
    # print(gt_list)
    # print("-----------------------------")
    # print(predict)
    # print("============================")

    print("current_image/total_iamge : {}/{}".format(current,total))
    print("the num of gt/the num of pred : {}/{} ".format(len(gt_labels), len(predict)))


    if ( miss_num == 1 ) :

        match_num += 1

        print("match_sum : {}".format(match_num )  )

        print("-----------------------------------------------------------")

        i = 0

        for i in range(len(gt_list)) : 

            gt = gt_list[i]

            gtcenter = ((gt[0]+gt[2]) / 2, (gt[1]+gt[3]) / 2 )

            # print(gtcenter)

            j = 0

            for j in range(len(predict)) :

                pr = predict[j]
                prcenter = (pr[0]+pr[2]) / 2, (pr[1]+pr[3]) / 2 

                distance = dist(gtcenter,prcenter)

                if j == 0 :
                    nearest = distance
                    match = j 
                    # print("----0-----")
                    # print(nearest)
                    # print(j)
                    # print(match)

                else :
                    if ( distance < nearest ) :
                        nearest = distance
                        match = j 
                        # print("----------")
                        # print(nearest)
                        # print(j)
                        # print(match)
                
            gt_labels[i].append(str(predict[match][4])+ "\n")

        #新建一個txt
        with open('datasets/bdd_val1/{}.txt'.format(name), 'w') as f:
            for i in range(len(gt_labels)) :

                for j in range(len(gt_labels[i])) :

                    f.write(gt_labels[i][j] )
                    if( j != 5 ) :
                        f.write(" ")

        shutil.copyfile(image_name, 'datasets/bdd_val1/{}.jpg'.format(name))

        # print(gt_labels)
        # plot_box_z(gt_labels, img )

                
    


    return match_num


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files, names = get_image_list(path)
    else:
        files = [path]
    files.sort()
    names.sort()
    total = len(names)
    current = 0
    match_num = 0
    for image_name, name in zip(files, names):

        current += 1

        # print(image_name)
        # print(name)


        gt_label = []
        gt_path = path + "/" + name + ".txt"

        f = None
        try:
            f = open(gt_path, 'r')
            for line in f.readlines():
                # line.replace("\n", "")
                # print(line)
                line = line.strip('\n')
                gt_label.append(line.split(" "))
                
        except IOError:
            print('ERROR: can not found ' + gt_path)
            if f:
                f.close()
        finally:
            if f:
                f.close()


        outputs, img_info = predictor.inference(image_name)
        print(image_name)


        if outputs != [None] :        
            img_z = img_info["raw_img"].copy()
            result_image, predict = predictor.visual(outputs[0], img_info, predictor.confthre)


            #在bdd dataset上加上我所偵測的z值
            match_num = add_z(img_z , gt_label, predict, match_num, current, total, image_name, name )




        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
