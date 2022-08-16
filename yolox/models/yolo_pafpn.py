#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[-1] * width), int(in_channels[-2] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[-2] * width),
            int(in_channels[-2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[-2] * width), int(in_channels[-3] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[-3] * width),
            int(in_channels[-3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[-3] * width), int(in_channels[-3] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[-3] * width),
            int(in_channels[-2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[-2] * width), int(in_channels[-2] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[-2] * width),
            int(in_channels[-1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    # ======= road segmentation ===============


        #fpn conv p3 
        self.bu_conv3 = Conv(
            int(in_channels[-2] * width), int(in_channels[-3] * width), 3, 1, act=act
        )

        #fpn conv p2 
        self.C3_p2 = CSPLayer(
            int(2 * in_channels[-4] * width),
            int(in_channels[-4] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.bu_conv4 = Conv(
            int(in_channels[-4] * width), int(in_channels[-5] * width), 3, 1, act=act
        )

        #fpn conv p1
        self.bu_conv5 = Conv(
            int(in_channels[-5] * width), int(in_channels[-6] * width), 3, 1, act=act
        )

        self.C3_p1 = CSPLayer(
            int(2 * in_channels[-7] * width),
            int(in_channels[-7] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )       


        #fpn segment head

        self.seghead_conv = Conv(
                    int(in_channels[-7] * width), 2, 3, 1, act=act
        )

    # ===================================================


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone

        # print(input.shape)
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        #pan可能是從這裡開始
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        


        # [ 16, Conv, [256, 128, 3, 1]],   #25
        # [ -1, Upsample, [None, 2, 'nearest']],  #26
        # [ -1, BottleneckCSP, [128, 64, 1, False]],  #27
        # [ -1, Conv, [64, 32, 3, 1]],    #28
        # [ -1, Upsample, [None, 2, 'nearest']],  #29
        # [ -1, Conv, [32, 16, 3, 1]],    #30
        # [ -1, BottleneckCSP, [16, 8, 1, False]],    #31
        # [ -1, Upsample, [None, 2, 'nearest']],  #32
        # [ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation head

        #segment接線的部分

        # print(pan_out2.shape)
        # print(f_out1.shape)
        fpn_out2 = self.bu_conv3(f_out1)
        f_out2 = self.upsample(fpn_out2)  # in:64,40,40  out:64,80,80	

        pan_out3 = self.C3_p2(f_out2)
        fpn_out3 = self.bu_conv4(pan_out3)
        f_out3 = self.upsample(fpn_out3)

        fpn_out4 = self.bu_conv5(f_out3)
        pan_out4 = self.C3_p1(fpn_out4)
        f_out4 = self.upsample(pan_out4)



        seg_output = self.seghead_conv(f_out4)
        
        outputs = (seg_output ,pan_out2, pan_out1, pan_out0)


        return outputs 
