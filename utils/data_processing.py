import os
import random
import xml.dom.minidom as xmldom
from shutil import copyfile, move

import cv2
from tqdm import tqdm
import h5py
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import torch

classes = {"person":1}

def voc2yolo(annotations_path, filename):
    """Convert PASCAL VOC annotation to YOLO annotation"""
    dom = xmldom.parse(os.path.join(annotations_path, filename))
    elements = dom.documentElement
    size = elements.getElementsByTagName("size")[0]
    w = int(size.getElementsByTagName("width")[0].firstChild.data)
    h = int(size.getElementsByTagName("height")[0].firstChild.data)
    object = elements.getElementsByTagName("object")
    data_list = []
    for i in range(len(object)):
        name = object[i].getElementsByTagName("name")[0].firstChild.data
        bndbox = object[i].getElementsByTagName("bndbox")[0]
        x1 = int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
        y1 = int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
        x2 = int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
        y2 = int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
        data = [classes["person"], (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
        data_list.append(data)
    return data_list
    # with open(os.path.join(r"D:\BUG\det_fuse\data\LLVIP\yolotxt",filename.split('.')[0]+".txt"), "w") as f:
    #     for item in data_list:
    #         f.write(" ".join([str(x) for x in item])+'\n')

# data = voc2yolo(r"260535.xml")
# files = os.listdir(r"D:\BUG\det_fuse\data\LLVIP\Annotations")
# for i in tqdm(range(len(files))):
#     voc2yolo(files[i])
            
def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def xywh2xyxy(label):
    x, y, w, h = label[1], label[2], label[3], label[4]
    x1 = int(x*640-0.5*w*640)
    y1 = int(y*640-0.5*h*640)
    x2 = int(x*640+0.5*w*640)
    y2 = int(y*640+0.5*h*640)
    return x1, y1, x2, y2

def get_target(labels):
    targets = []
    for label in labels:
        boxes = []
        labels = []
        for item in label:
            target = dict()
            x1, y1, x2, y2 = xywh2xyxy(item)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(item[0]))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        targets.append(target)
    return targets

def divide_data(vis_path, ir_path):
    # VIS_fils = sorted(get_img_file(blur_path))
    # for filename in VIS_fils:
    #     f = os.path.join(ir_path, filename.split("\\")[-1])
    #     copyfile(f, os.path.join(target_path, filename.split("\\")[-1]))
    VIS_fils =get_img_file(vis_path)
    random.shuffle(VIS_fils)
    # IR_files = sorted(get_img_file(ir_path))
    train_num = int(len(VIS_fils) * 0.8)
    print(train_num)
    for i in range(train_num):
        move(VIS_fils[i], os.path.join(vis_path, "train", VIS_fils[i].split("\\")[-1]))
        move(os.path.join(ir_path, VIS_fils[i].split("\\")[-1]), os.path.join(ir_path, "train", VIS_fils[i].split("\\")[-1]))
    
    for i in range(train_num+1, len(VIS_fils)):
        move(VIS_fils[i], os.path.join(vis_path, "test", VIS_fils[i].split("\\")[-1]))
        move(os.path.join(ir_path, VIS_fils[i].split("\\")[-1]), os.path.join(ir_path, "test", VIS_fils[i].split("\\")[-1]))
    

def create_h5(vis_path, ir_path, mode='train'):
    data_name = "LLVIP_blur_{}".format(mode)
    imsize = 640
    # train_IR_files = sorted(get_img_file(r"D:\BUG\det_fuse\data\LLVIP\infrared\train"))
    # train_VIS_files   = sorted(get_img_file(r"D:\BUG\det_fuse\data\LLVIP\visible\train"))
    IR_files = sorted(get_img_file(os.path.join(ir_path, mode)))
    VIS_files   = sorted(get_img_file(os.path.join(vis_path, mode)))
    assert len(IR_files) == len(VIS_files)
    h5 = h5py.File(os.path.join(r'data', data_name+'.h5'), 'w')
    h5_ir = h5.create_group('ir')
    h5_vis = h5.create_group('vis')
    h5_label = h5.create_group('label')
    # h5_ir.create_dataset("train", (train_num, imsize, imsize))
    # h5_vis.create_dataset("train", (train_num, imsize, imsize))
    label_data = np.array([])
    for i in tqdm(range(len(IR_files))):
        assert IR_files[i].split('\\')[-1].split('.')[0] == VIS_files[i].split('\\')[-1].split('.')[0]
        I_IR = resize(rgb2gray(imread(IR_files[i])), (imsize, imsize)) # [1, H, W] Float32    
        h5_ir.create_dataset(str(i), data=I_IR)
        # h5_ir["train"][i] = I_IR
        I_VIS = resize(rgb2gray(imread(VIS_files[i])), (imsize, imsize)) # [1, H, W] Float32
        h5_vis.create_dataset(str(i), data=I_VIS)
        # h5_vis["train"][i] = I_VIS
        label = np.array(voc2yolo(r"D:\EdgeDownload\data\LLVIP\Annotations", IR_files[i].split('\\')[-1].split('.')[0]+'.xml')) # [class, x, y, w, h]
        h5_label.create_dataset(str(i), data = label)
        # np.append(label_data, label)
    # h5_label.create_dataset("train", data = label_data)
    h5.close()


if __name__ == "__main__":
    divide_data(r"D:\EdgeDownload\data\LLVIP\blur\vis", r"D:\EdgeDownload\data\LLVIP\blur\ir")
    # create_h5(r"D:\EdgeDownload\data\LLVIP\blur\vis", r"D:\EdgeDownload\data\LLVIP\blur\ir", 'train')
    # create_h5(r"D:\EdgeDownload\data\LLVIP\blur\vis", r"D:\EdgeDownload\data\LLVIP\blur\ir", 'test')
    # with h5py.File(os.path.join(r'data', "LLVIP_train"+'.h5'),"r") as f:
    #     for key in f.keys():
    #         print(f[key], key, f[key].name)
