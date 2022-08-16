#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import sys

from matplotlib.pyplot import draw_if_interactive
sys.path.remove('/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOX')
sys.path.append("/home/lab602.10977014_0n1/.pipeline/10977014/YOLOXP/")

import cv2

import torch
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, vis_z
from yolox.evaluators.evaluate import ConfusionMatrix,SegmentationMetric
import torchvision.transforms as transforms
from yolox.data import COCODataset
import math

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "-demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-segs", default=True, help="draw segmentation",
                        action="store_true")

    parser.add_argument(
        "--path", default="datasets/bddseg/val2017", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=False,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/custom/yolox_s_seg.py",
        type=str,
        help="pls input your experiment description file",
    )
    ###############################################################記得修改下面的解析度 
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_s_seg_paper/latest_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.1, type=float, help="test nms threshold")
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


def load_cocodataset() :

    data = COCODataset(
        data_dir="datasets/bddseg",
        json_file="instances_val2017.json",
        name="val2017",
        roadgt="valroad",
        # img_size=(288, 512),
        img_size=(576, 1024),
        preproc=None,
        cache=False,
        is_train=False
    )
    return data

def get_image_gt(image_names, data) :
    
    # res, img_info, resized_info, file_name = data.pull_item(image_names)
    # print(data.json_file)
    # res, img_info, resized_info, _ = data.annotations[image_names-1]
    res, img_info, resized_info, file_name, _=data.load_anno_from_ids(int(image_names))
    # print("img_info" +str(img_info))
    # print("-------get_image_gt---------")
    # print(res)
    # print("-------get_image_gt---------")
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageMeter_forz(object):

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0




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
        segs=False
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.img_channel = exp.img_channel
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.segs = segs
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
            if self.img_channel == 4:
                img = cv2.imread(img, -1)
            else:
                img = cv2.imread(img)
        else:
            img_info["file_name"] = None



        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])

    
        img_info["ratio"] = ratio


   
        img, _, _ = self.preproc(img, None, self.test_size, None )


        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs, seg_output = self.model(img)



            # print("first seg_output :{}".format(seg_output.shape))

            if self.decoder is not None:  # None
                outputs, seg_output = self.decoder(outputs, seg_output, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, seg_output, img_info

    def visual(self, output, seg_output, img_info, cls_conf=0.35, draw_seg=False):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, np.zeros_like(img)
        output = output.cpu()
        bboxes = output[:, 0:4]


        # preprocessing: resize
        #(640,384) -> (1280,720)

        
        # print(ratio)
        bboxes /= ratio

        
        # img = cv2.resize(img,(640,384),interpolation=cv2.INTER_LINEAR)

        # kps = output[:, 7:] / ratio if draw_kp else []
        if draw_seg:

            # print("img.shape :{}".format(img.shape))

            


            seg = seg_output.max(axis=0)[1].cpu().numpy()

            h, w, _ = img.shape
            sh, sw = seg.shape

            # print(seg.shape)

 

            # print((int(sw / ratio), int(sh / ratio)))


            
            #preprocessing : bbboxes resize
            bbox_h = sh / h 
            bbox_w = sw / w


            # print("=============")
            # print(bbox_h)
            # print(bbox_w)

            # for box in bboxes : 
            #     box[0] /= 0.49
            #     box[1] /= bbox_w
            #     box[2] /= 0.49
            #     box[3] /= bbox_w 






            # seg = cv2.resize(
            #     seg, (int(sw / ratio), int(sh / ratio)),
            #     interpolation=cv2.INTER_NEAREST)[:h, :w]

            seg = cv2.resize(
                seg, (int(sw / ratio), int(sh / ratio)),
                interpolation=cv2.INTER_NEAREST)[:h, :w]




        else:
            seg = []
        z = output[:,4]
        cls = output[:, 7]
        scores = output[:, 5] * output[:, 6]

        # img_id = img_info["file_name"].split('.')[0]        
        # gt = get_image_gt(int(img_id)) 
        vis_res, seg_mask = vis(img, bboxes, scores, z ,cls, cls_conf, self.cls_names, seg)


        # gt_bboxes = gt[:, 0:4]
        # gt_bboxes /= ratio
        # gt_z = gt[:,4]
        # vis_res_gt = vis_z(vis_res,gt_bboxes,gt_z)

        return vis_res , seg_mask
        # return vis_res_gt , seg_mask


def get_seggt( mask_path, shape ) :

    _, _, h, w = shape


    seg_label = cv2.imread(str(mask_path), 0)

    # if r != 1:  # always resize down, only resize up if training with augmentation	
    #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR	

    seg_label = cv2.resize(seg_label, (int(w), int(h)), interpolation=cv2.INTER_AREA)	

    _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
    _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)

    seg1 = transforms.ToTensor()(seg1)
    seg2 = transforms.ToTensor()(seg2)

    seg_label = np.zeros((h, w, 1))
    seg_label = torch.stack((seg2[0], seg1[0]),0)

    seg_targets = []

    seg_targets.append(seg_label.unsqueeze(0))
    seg_targets= torch.cat(seg_targets, 0)



    return seg_targets

def transfer_outputbox( boxes, z ) :


    addz_boxes = []

    z = z.numpy()
    

    for i in range(len(boxes)) :
        box = boxes[i]
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]
        z0 = z[i]



        addz_boxes.append([int(x0), int(y0), int(x1), int(y1), z0])


    return addz_boxes

def transfer_gtbox(boxes) :

 

    new_boxes = []
    for i in range(len(boxes)) :
        box = boxes[i]

        x0 = float(box[0]) 
        y0 = float(box[1]) 
        x1 = float(box[2]) 
        y1 = float(box[3])
        z = float(box[4])


        new_boxes.append([int(x0), int(y0), int(x1), int(y1), z])

    
    return new_boxes






def plot_box(boxes, img, color=None , line_thickness=None):

 
    height, width, _ = img.shape 
    

    for i in range(len(boxes)) :
        box = boxes[i]
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]
        
        tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thicknesss
        # color = color or [random.randint(0, 255) for _ in range(3)]
        # color = (255,0,0)
        c1, c2 = (int(x0),int(y0)), (int(x1),int(y1))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # cv2.imwrite("./datasets/gtimg.jpg", img)


    return img

def dist(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

def image_demo(predictor, vis_folder, path, current_time, save_result, draw_seg):



    data = load_cocodataset()


    da_metric = SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    z_deviation = AverageMeter_forz()

    if os.path.isdir(path):
        files, names = get_image_list(path)
    else:
        files = [path]
    files.sort() 
    names.sort()
    for image_name, name in zip(files, names):





        outputs, seg_outputs, img_info = predictor.inference(image_name)
        _,da_predict=torch.max(seg_outputs, 1)

        if seg_outputs is None:
            seg_outputs = [None for _ in range(len(outputs))]
        
        result_image, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info,
                                        predictor.confthre, draw_seg)



        seg_path = path.replace("2017","road")
        da_gt = get_seggt( seg_path + "/" + name + ".png", seg_outputs.shape )
        _,da_gt=torch.max(da_gt, 1)


        if (seg_outputs[0] != None) :


            # driving area segment evaluation     
            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,1)
            da_IoU_seg.update(da_IoU,1)
            da_mIoU_seg.update(da_mIoU,1)
            # print(seg_mask.shape)
        

        if (outputs[0] != None) :
            # z depth evalutaion 
            res = get_image_gt(name, data)
            ratio = img_info["ratio"]
            output_z = outputs[0].cpu()
            bboxes = output_z[:, 0:4]/ratio
            zs = output_z[:,4]
            res[:,:4] = res[:,:4] / ratio
            
            addbox = transfer_outputbox( bboxes, zs )
            # print(addbox)
            addres =transfer_gtbox( res )
            # print(addres)
            all_zoff = 0

            z_sum = len(addbox)
            for i in range(len(addbox)) : 

                pr = addbox[i]
                prcenter = (pr[0]+pr[2]) / 2, (pr[1]+pr[3]) / 2 

                # print(gtcenter)

                j = 0

                for j in range(len(addres)) :

                    gt = addres[j]

                    gtcenter = ((gt[0]+gt[2]) / 2, (gt[1]+gt[3]) / 2 )

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
                

                z_off = abs(addres[match][4] - pr[4])

                all_zoff += z_off

                print("z_sum : {}".format(z_sum))
                print("z_off : {}".format(z_off))
                    

            z_deviation.update(all_zoff, z_sum)







        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            if draw_seg:
                if '.jpg' in save_file_name:
                    cv2.imwrite(save_file_name.replace('.jpg', '_seg.jpg'), seg_mask)
                else:
                    cv2.imwrite(save_file_name.replace('.png', '_seg.png'), seg_mask)



        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
            break

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)

    print("da_segment_result : {}".format(da_segment_result))
    print("z_deviation.avg : {}".format(z_deviation.avg))
    print("z_deviation.sum : {}".format(z_deviation.sum))
    


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        
        if ret_val:
            # frame = cv2.resize(frame, (1392,512))	
            outputs, seg_outputs, img_info = predictor.inference(frame)
            result_frame, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info, predictor.confthre, True)
            # result_frame = cv2.resize(result_frame, (1920, 1080))	
            if args.save_result:
                vid_writer.write(result_frame)
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
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size, exp.img_channel)))

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

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device,
                          args.fp16, args.legacy )

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result,
                   args.segs)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
