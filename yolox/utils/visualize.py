#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis","vis_z"]


def label2image(pred, COLORMAP):	
    COLORMAP[0] = np.array([0, 0, 0])	
    colormap = np.array(COLORMAP * 255, dtype='uint8')	
    X = pred.astype('int32')	
    return colormap[X]

def vis(img, boxes, scores, z, cls_ids, conf=0.5, class_names=None, seg=[]):
    # print(cls_ids)
    # print(int(cls_ids[0]))
    # print("------")
    # print(len(boxes))
    # print("------")


    for i in range(len(boxes)):

        box = boxes[i]
        # print(box)
        cls_id = int(cls_ids[i])
        az = float( z[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        if cls_id == 0 :
            color = [255, 255, 0]

        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100 )
        # text = text + ' {}:{:.1f}m'.format("D",az)
        text_z = '{:.1f}m'.format(az)



        # seg vis
        if len(seg) > 0:	

                h, w, channel = img.shape	

                if channel == 4:	
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)	
                seg_mask = label2image(seg, _COLORS)[:, :, ::-1]	
                for ii, vv in enumerate(np.unique(seg)):	
                    if vv == 0:	
                        continue	
                    img[seg == vv] = img[seg == vv] * 0.3 + _COLORS[ii] * 255 * 0.7	
        else:	
            seg_mask = np.zeros_like(img)	


        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)



    
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # print("txt_size :")
        # print(txt_size)
        # txt_size_z = cv2.getTextSize(text_z, font, 0.4, 1)[0]
        boxwidth = abs(x1-x0)

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()

        # blk = np.zeros(img.shape, np.uint8)  
        # cv2.rectangle(
        #     blk,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)



        # cv2.circle(img, (int((x0+x1)/2),int((y0+y1)/2)), radius = 5, color = (0,0,255), thickness = 0.5 )
        

        if (boxwidth < 100 ) :

            
            txt_size = cv2.getTextSize(text, font, 0.4, 2)[0]
            txt_z_size = cv2.getTextSize(text_z,font, 0.4, 2 )[0]
            
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, 2, cv2.LINE_AA)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), 0, cv2.LINE_AA)
            # cv2.circle(img, (int((x0+x1)/2),int((y0+y1)/2)), radius = 1, color = (0,0,255), thickness = -1  )
            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 0.4, (0,0,255), 2, cv2.LINE_AA )


            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 0.4, (0, 0, 0), 0, cv2.LINE_AA )
        elif(boxwidth >= 100 and boxwidth < 200) :
            txt_size = cv2.getTextSize(text, font, 0.6, 3)[0]
            txt_z_size = cv2.getTextSize(text_z,font, 0.8, 4 )[0]
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, 3, cv2.LINE_AA)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, (0, 0, 0), 0, cv2.LINE_AA)
            # cv2.circle(img, (int((x0+x1)/2),int((y0+y1)/2)), radius = 3, color = (0,0,255), thickness = -1  )
            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 0.8, (0,0,255), 4, cv2.LINE_AA )
            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA )
        elif(boxwidth >= 200 ) :
            txt_size = cv2.getTextSize(text, font, 0.8, 4)[0]
            txt_z_size = cv2.getTextSize(text_z,font, 1.2, 6 )[0]
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.8, txt_color, 4, cv2.LINE_AA)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.circle(img, (int((x0+x1)/2),int((y0+y1)/2)), radius = 5, color = (0,0,255), thickness = -1  )
            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 1.2, (0,0,255), 6, cv2.LINE_AA )
            cv2.putText(img, text_z, (int((x0+x1)/2-(txt_z_size[0]/2)), int((y0+y1)/2)), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA )





        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] +txt_size_z[0] + 1, y0 + int(1.5*(txt_size[1]+ txt_size_z[1]))),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text_z, (x0, y0 + txt_size_z[1]), font, 0.4, txt_color, thickness=1)


    return img, seg_mask

def vis_z(img, boxes, z ):

    for i in range(len(boxes)):
        box = boxes[i]
        # print(box)
        az = float(z[i])
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        # color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100 )
        text = ' {}:{:.1f}m'.format("D",az)
        
        # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font,0.4, 1)[0]
        # txt_size_z = cv2.getTextSize(text_z, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 1)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()

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



    return img

_COLORS = np.array(
    [
        0.494, 0.184, 0.556,
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125

    ]
).astype(np.float32).reshape(-1, 3)
# _COLORS = np.array(
#     [
#         0.000, 0.447, 0.741,
#         0.850, 0.325, 0.098,
#         0.929, 0.694, 0.125,
#         0.494, 0.184, 0.556,
#         0.466, 0.674, 0.188,
#         0.301, 0.745, 0.933,
#         0.635, 0.078, 0.184,
#         0.300, 0.300, 0.300,
#         0.600, 0.600, 0.600,
#         1.000, 0.000, 0.000,
#         1.000, 0.500, 0.000,
#         0.749, 0.749, 0.000,
#         0.000, 1.000, 0.000,
#         0.000, 0.000, 1.000,
#         0.667, 0.000, 1.000,
#         0.333, 0.333, 0.000,
#         0.333, 0.667, 0.000,
#         0.333, 1.000, 0.000,
#         0.667, 0.333, 0.000,
#         0.667, 0.667, 0.000,
#         0.667, 1.000, 0.000,
#         1.000, 0.333, 0.000,
#         1.000, 0.667, 0.000,
#         1.000, 1.000, 0.000,
#         0.000, 0.333, 0.500,
#         0.000, 0.667, 0.500,
#         0.000, 1.000, 0.500,
#         0.333, 0.000, 0.500,
#         0.333, 0.333, 0.500,
#         0.333, 0.667, 0.500,
#         0.333, 1.000, 0.500,
#         0.667, 0.000, 0.500,
#         0.667, 0.333, 0.500,
#         0.667, 0.667, 0.500,
#         0.667, 1.000, 0.500,
#         1.000, 0.000, 0.500,
#         1.000, 0.333, 0.500,
#         1.000, 0.667, 0.500,
#         1.000, 1.000, 0.500,
#         0.000, 0.333, 1.000,
#         0.000, 0.667, 1.000,
#         0.000, 1.000, 1.000,
#         0.333, 0.000, 1.000,
#         0.333, 0.333, 1.000,
#         0.333, 0.667, 1.000,
#         0.333, 1.000, 1.000,
#         0.667, 0.000, 1.000,
#         0.667, 0.333, 1.000,
#         0.667, 0.667, 1.000,
#         0.667, 1.000, 1.000,
#         1.000, 0.000, 1.000,
#         1.000, 0.333, 1.000,
#         1.000, 0.667, 1.000,
#         0.333, 0.000, 0.000,
#         0.500, 0.000, 0.000,
#         0.667, 0.000, 0.000,
#         0.833, 0.000, 0.000,
#         1.000, 0.000, 0.000,
#         0.000, 0.167, 0.000,
#         0.000, 0.333, 0.000,
#         0.000, 0.500, 0.000,
#         0.000, 0.667, 0.000,
#         0.000, 0.833, 0.000,
#         0.000, 1.000, 0.000,
#         0.000, 0.000, 0.167,
#         0.000, 0.000, 0.333,
#         0.000, 0.000, 0.500,
#         0.000, 0.000, 0.667,
#         0.000, 0.000, 0.833,
#         0.000, 0.000, 1.000,
#         0.000, 0.000, 0.000,
#         0.143, 0.143, 0.143,
#         0.286, 0.286, 0.286,
#         0.429, 0.429, 0.429,
#         0.571, 0.571, 0.571,
#         0.714, 0.714, 0.714,
#         0.857, 0.857, 0.857,
#         0.000, 0.447, 0.741,
#         0.314, 0.717, 0.741,
#         0.50, 0.5, 0
#     ]
# ).astype(np.float32).reshape(-1, 3)
