import numpy as np
from math import sqrt as sqrt
from itertools import product as product
import matplotlib.pyplot as plt
from ipdb import set_trace
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections
import torch

# 中心坐标转换为角点坐标
def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

# 创建初始框
def create_prior_boxes():
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    fmap_dims = collections.OrderedDict()
    fmap_dims['conv4_3'] = 38
    fmap_dims['conv7'] = 19
    fmap_dims['conv8_2'] = 10
    fmap_dims['conv9_2'] = 5
    fmap_dims['conv10_2'] = 3
    fmap_dims['conv11_2'] = 1

    obj_scales = collections.OrderedDict()
    obj_scales['conv4_3'] = 0.1
    obj_scales['conv7'] = 0.2
    obj_scales['conv8_2'] = 0.375
    obj_scales['conv9_2'] = 0.55
    obj_scales['conv10_2'] = 0.725
    obj_scales['conv11_2'] = 0.9

    aspect_ratios = collections.OrderedDict()
    aspect_ratios['conv4_3'] =  [1., 2., 0.5]
    aspect_ratios['conv7'] = [1., 2., 3., 0.5, .333]
    aspect_ratios['conv8_2'] = [1., 2., 3., 0.5, .333]
    aspect_ratios['conv9_2'] =  [1., 2., 3., 0.5, .333]
    aspect_ratios['conv10_2'] = [1., 2., 0.5]
    aspect_ratios['conv11_2'] = [1., 2., 0.5]

    fmaps = list(fmap_dims.keys())
    prior_boxes = []
    feature_box = {}
    for k, fmap in enumerate(fmaps):
        feature_box[fmap] = []
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                    feature_box[fmap].append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])
                        feature_box[fmap].append([cx, cy, additional_scale, additional_scale])
    for key in feature_box.keys():
        feature_box[key] = torch.FloatTensor(feature_box[key])
        feature_box[key].clamp_(0, 1)
        feature_box[key] = cxcy_to_xy(feature_box[key])
        feature_box[key] = feature_box[key].numpy()*1024
    return feature_box

# 将每个特征层产生的初始框画在原图上
def plot_all(feature_box):
    for feature in feature_box:
        print(feature)
        img_PIL = Image.open('test.jpg')
        for i  in range(len(feature_box[feature])):
            xmin = int(feature_box[feature][i, 0])
            ymin = int(feature_box[feature][i, 1])
            xmax = int(feature_box[feature][i, 2])
            ymax = int(feature_box[feature][i, 3])
            # for i in range(3):
            b=ImageDraw.ImageDraw(img_PIL)
            b.rectangle((xmin+i,ymin+i,xmax-i,ymax-i),fill=None,outline='red')
        img_PIL.show()
        img_PIL.save(feature+'_prior_boxes_all.png')
        # set_trace()

# 单个框显示
def plot_single_group(feature_box):
    for feature  in feature_box:
        img_PIL = Image.open('test.jpg')
        if feature != 'conv8_2':
            for i in range(len(feature_box[feature])):
                xmin = int(feature_box[feature][i, 0])
                ymin = int(feature_box[feature][i, 1])
                xmax = int(feature_box[feature][i, 2])
                ymax = int(feature_box[feature][i, 3])
                # 根据特殊图像，特殊处理conv4_3使之定位到目标
                # if i>115 and i<120:
                #     # print(feature_box['conv4_3'][i])    
                #     for k in range(3):
                #         b=ImageDraw.ImageDraw(img_PIL)
                #         b.rectangle((xmin+k,ymin+k,xmax-k,ymax-k),fill=None,outline='red')
                if abs(((xmin+xmax)/2)-512)<14 and abs(((ymin+ymax)/2)-512)<14 :
                    print(abs(((xmin+xmax)/2)-512))
                    for j in range(3):
                        b=ImageDraw.ImageDraw(img_PIL)
                        b.rectangle((xmin+j,ymin+j,xmax-j,ymax-j),fill=None,outline='red')
                        print(xmin,ymin,xmax,ymax)
        else:
            for i in range(len(feature_box[feature])):
                xmin = int(feature_box[feature][i, 0])
                ymin = int(feature_box[feature][i, 1])
                xmax = int(feature_box[feature][i, 2])
                ymax = int(feature_box[feature][i, 3])
                if abs(((xmin+xmax)/2)-512)<51 and abs(((ymin+ymax)/2)-512)<51 :
                    print(abs(((xmin+xmax)/2)-512))
                    for j in range(3):
                        b=ImageDraw.ImageDraw(img_PIL)
                        b.rectangle((xmin+j,ymin+j,xmax-j,ymax-j),fill=None,outline='red')
                        print(xmin,ymin,xmax,ymax)

        img_PIL.show()
        img_PIL.save(feature+'.png')


def main():
    # 创建初始框
    feature_box = create_prior_boxes()

    # 画出每个特征图上的所有初始框
    plot_all(feature_box)

    # 每个特征图上画一组初始框
    plot_single_group(feature_box)




    
if __name__ == '__main__':
    main()