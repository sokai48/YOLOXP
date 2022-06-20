import numpy as np
import os
import cv2
import kitti_util as utils

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        # print(root_dir, split)
        # self.split_dir = os.path.join(root_dir, split)
        self.split_dir = os.path.join(root_dir)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 0
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image")
        self.label_dir = os.path.join(self.split_dir, "labels")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    # def get_image(self, idx):
    #     assert idx < self.num_samples
    #     img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
    #     return utils.load_image(img_filename)

    # def get_lidar(self, idx, dtype=np.float32, n_vec=4):
    #     assert idx < self.num_samples
    #     lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
    #     print(lidar_filename)
    #     return utils.load_velo_scan(lidar_filename, dtype, n_vec)
    def get_label_objects(self, idx):
        # assert idx < self.num_samples and self.split == "training"
        # print("%06d.txt" % (idx))
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        # assert 1==0, f"{utils.read_label(label_filename)}"
        return utils.read_label(label_filename)

    def convert_to_yolo(self, size, box) : 

        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh


        return (x,y,w,h)

if __name__ == "__main__":
    # import mayavi.mlab as mlab
    # from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    #抓出原始圖片大小
    img = cv2.imread("/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXZ/datasets/004076.png")
    oh,ow = img.shape[0], img.shape[1]


    #讀入原始label
    root_dir = "/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOX/YOLO2COCO/dataset/kitti/stamp/"
    output_dir = "/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOX/YOLO2COCO/dataset/kitti/stamp/newlabels/"
    dataset = kitti_object(root_dir)
    # print(len(dataset))
    for data_idx in range(len(dataset)):
        # print(data_idx)
        objects = dataset.get_label_objects(data_idx)

        list = []

        #轉換lable_3d_to_2d

        # path = output_dir + ("%06d.txt" % (data_idx))
        # f = open(path,'w')

        for obj in objects:

            if obj.type == "Car" or obj.type == "Van" or obj.type =="Pedestrian" or obj.type =="Person_sitting" or obj.type =="Cyclist" :

                if obj.type == "Car"  or obj.type =="Van" : type = 0
                elif obj.type == "Pedestrian" or obj.type =="Person_sitting" : type = 1
                elif obj.type == "Cyclist" : type = 2

                print(type)
                box = (obj.xmin,obj.xmax,obj.ymin,obj.ymax)
                x,y,w,h = dataset.convert_to_yolo( (ow,oh), box )
                z = obj.t[2]

                column = [str(type),' ',str(x),' ',str(y),' ',str(w),' ',str(h),' ',str(z),'\n']


                # list.append(column)
                print(column)
                # f.writelines(column)
                # print(list)

        # with open ('N_a.txt','w') as q:
        # for i in a:
        #     for e in range(len(a[0])):
        #         t=t+str(i[e])+' '
        #     q.write(t.strip(' '))
        #     q.write('\n')
        #     t=''

    
        f.close


        #輸出成需要的txt