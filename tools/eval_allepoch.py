#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import sys
sys.path.append("/home/lab602.10977014_0n1/.pipeline/10977014/YOLOXP/")

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger

# import matplotlib相關套件
import matplotlib.pyplot as plt

# import字型管理套件
from matplotlib.font_manager import FontProperties


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-seg", help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_s_seg_5000_178epoch/", type=str, help="ckpt for eval")
    parser.add_argument("--conf", default="0.001", type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
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
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        # if args.ckpt is None:
        #     ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        # else:
        ckpt_root = args.ckpt

    # for ckpt_file in os.listdir(ckpt_root) :
    
    path_list = os.listdir(ckpt_root)
    path_list.sort(key = lambda x: (len(x),x))

    ap_list = []
    ar_list = []

    epoch_num = 0
    epoch_name = []
    
    for filename in path_list :   
        if filename[:5] == "epoch" :
            epoch_num += 10
            epoch_name.append(epoch_num)
            ckpt_file = os.path.join(ckpt_root, filename)
            print(ckpt_file)
            logger.info("loading checkpoint from {}".format(ckpt_file))
            loc = "cuda:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc)
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

            if is_distributed:
                model = DDP(model, device_ids=[rank])

            if args.fuse:
                logger.info("\tFusing model...")
                model = fuse_model(model)

            if args.trt:
                assert (
                    not args.fuse and not is_distributed and args.batch_size == 1
                ), "TensorRT model is not support model fusing and distributed inferencing!"
                trt_file = os.path.join(file_name, "model_trt.pth")
                assert os.path.exists(
                    trt_file
                ), "TensorRT model is not found!\n Run tools/trt.py first!"
                model.head.decode_in_inference = False
                decoder = model.head.decode_outputs
            else:
                trt_file = None
                decoder = None

            # start evaluate
            *_, summary, ap, ar = evaluator.evaluate(
                model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
            )

            print(summary)

            ap_list.append(ap)
            ar_list.append(ar)
            logger.info("\n" + summary)


 
    ap_0 = []
    ap_1 = []
    ap_2 = []
    map = []
    ar_0 = []
    ar_1 = []
    ar_2 = [] 

    for i in range(len(ap_list)) :
        ap_0.append(ap_list[i]['0'])
        ap_1.append(ap_list[i]['1'])
        ap_2.append(ap_list[i]['2'])
        map.append((ap_list[i]['0']+ ap_list[i]['1'] + ap_list[i]['2'])/3)
        ar_0.append(ap_list[i]['0'])
        ar_1.append(ap_list[i]['1'])
        ar_2.append(ap_list[i]['2'])
        # ap_2
        # ar_0
        # ar_1
        # ar_2
    plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
    plt.plot(epoch_name,ap_0,'s-',color = 'b', label="Car")
    plt.plot( epoch_name,ap_1,'o-',color = 'g', label="Pesterian")
    plt.plot( epoch_name,ap_2,'x-',color = 'y', label="Cyclist")
    plt.plot( epoch_name,map,'d-',color = 'r', label="map")
    plt.title("ap of every epoch", fontsize=30, x=0.5, y=1.03)

    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 標示x軸(labelpad代表與圖片的距離)
    plt.xlabel("epoch", fontsize=30, labelpad = 15)

    # 標示y軸(labelpad代表與圖片的距離)
    plt.ylabel("ap_per_class(%)", fontsize=30, labelpad = 20)

    # 顯示出線條標記位置
    plt.legend(loc = "best", fontsize=20)

    plt.savefig('evaluate_output/plot.png')

    
            


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
