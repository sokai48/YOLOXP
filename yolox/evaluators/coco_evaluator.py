#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch


from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

# from yolox.utils.metric2 import ConfusionMatrix,ap_per_class,box_iou

# def process_batch(detections, labels, iouv):
#     """
#     Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
#     Arguments:
#         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#         labels (Array[M, 5]), class, x1, y1, x2, y2
#     Returns:
#         correct (Array[N, 10]), for 10 IoU levels
#     """
#     correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
#     iou = box_iou(labels[:, 1:], detections[:, :4])
#     correct_class = labels[:, 0:1] == detections[:, 5]
#     for i in range(len(iouv)):
#         x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 # matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#             correct[matches[:, 1].astype(int), i] = True
#     return correct

def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table, per_class_AR


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    # print(precisions)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair] 
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )


    return table, per_class_AP


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        # zthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
        segcls = 2
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        # self.zthre = zthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.segcls = segcls

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        #from v5 https://zhuanlan.zhihu.com/p/499759736
        # iouv = torch.linspace(0.5, 0.95, 10, device='cpu')
        # niou = iouv.numel()
        # confusion_matrix = ConfusionMatrix(nc=self.num_classes)
        # stats=[]
        # seen=0
        # names=["Car","Pedestrian","Cyclist"] #类名
        # names_dic=dict(enumerate(names)) #类名字典
        # # print(names_dic)
        # s = ('\n%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')    
        # save_dir="evaluate_output"]
        #from v5 https://zhuanlan.zhihu.com/p/499759736
        """
        from yolox.evaluators.evaluate import ConfusionMatrix,SegmentationMetric
        da_metric = SegmentationMetric(self.seg_cls) #segment confusion matrix 
        da_acc_seg = AverageMeter()
        da_IoU_seg = AverageMeter()
        da_mIoU_seg = AverageMeter()
        """

        print(enumerate(progress_bar(self.dataloader)))
        print(self.dataloader.dataset.coco)
        for cur_iter, (imgs, _, info_imgs, ids, _) in enumerate(
            progress_bar(self.dataloader)
        ):
            print(imgs.shape)

            print(info_imgs)

            print(ids)

            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs, seg_outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

                print(outputs)
                print(seg_outputs)

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

            """
            #driving area segment evaluation
            _,da_predict=torch.max(seg_outputs, 1)
            _,da_gt=torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,imgs.size(0))
            da_IoU_seg.update(da_IoU,imgs.size(0))
            da_mIoU_seg.update(da_mIoU,imgs.size(0))      

            """      

            #from v5 https://zhuanlan.zhihu.com/p/499759736
            # for _id,out in zip(ids,outputs):
            #     seen += 1
            #     gtAnn=self.dataloader.dataset.coco.imgToAnns[int(_id)]
            #     tcls=[(((its["category_id"])-1))for its in gtAnn]
            #     print(tcls)
            #     if out==None: 
            #         stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            #         continue
            #     else:
            #         gt=torch.tensor([[(its['category_id'])]+its['clean_bbox'] for its in gtAnn])
            #         dt=out.cpu().numpy()
            #         # print(dt)
            #         dt[:,5]=dt[:,5]*dt[:,6]
            #         dt[:,6]=dt[:,7]
            #         dt=torch.from_numpy(np.delete(dt,-1,axis=1))#share mem
            #         confusion_matrix.process_batch(dt, gt)
            #         correct = process_batch(dt, gt, iouv)
            #         stats.append((correct, dt[:, 5], dt[:, 6], tcls))
            #from v5 https://zhuanlan.zhihu.com/p/499759736


        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        #from v5 https://zhuanlan.zhihu.com/p/499759736
        # stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # tp, fp, p, r, f1, ap, ap_class =ap_per_class(*stats, plot=True, save_dir=save_dir, names=names_dic)
        # confusion_matrix.plot(save_dir=save_dir, names=names)
        # ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_classes)
        # pf = '\n%20s' + '%11i'  *2 + '%11.3g' * 4  # print format
        # s+=pf % ('all',seen, nt.sum(), mp, mr, map50, map)
        # for i, c in enumerate(ap_class):
        #     s+=pf % (names[c],seen, nt[c], p[i], r[i], ap50[i], ap[i])
        # logger.info(s)
        #from v5 https://zhuanlan.zhihu.com/p/499759736

        print(eval_results)

        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            # print(output.shape)
            z = output[:, 4]
            cls = output[:, 7]
            scores = output[:, 5] * output[:, 6]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "z": z[ind].numpy().item(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            print(cocoGt)
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")



            print(cocoGt)
            print(cocoDt)
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table, AP_per_class = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table, AR_per_class = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"

            # print(AP_table["class"])
            # print(AP_table)
            # print(cocoEval.stats[0], cocoEval.stats[1], info)
            # print("===============")
            # print(cocoEval.stats)
            # print("--------------")
            # print(info)


            # return cocoEval.stats[0], cocoEval.stats[1], info, AP_per_class, AR_per_class
            return cocoEval.stats[0], cocoEval.stats[1], info 
        else:
            # return 0, 0, info, [], []
            return 0, 0, info
           


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
