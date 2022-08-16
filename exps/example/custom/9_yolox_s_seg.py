#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/bddseg"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 4

        self.max_epoch = 200
        self.data_num_workers = 0
        self.eval_interval = 1

        # self.input_size = (375, 1242) 
        # self.input_size = (192, 640)
        # self.test_size = (192, 640)

        # self.input_size = (720,1280)
        # self.test_size = (720,1280)

        # 1280 720
        # 768 448
        # 640 360
        #576 1024
        #288 512

        # self.input_size = (576, 1024) 
        # self.test_size = (576, 1024)

        self.input_size = (288, 512) 
        self.test_size = (288, 512) 


        self.img_channel = 3

        self.eval_interval = 10

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0
        # prob of applying mixup aug
        self.mixup_prob = 0
        # prob of applying hsv aug
        self.hsv_prob = 0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0
        self.mosaic_scale = 0
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = 0
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 0


        