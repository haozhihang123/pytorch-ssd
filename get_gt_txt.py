#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import json
import xml.etree.ElementTree as ET
from ipdb import set_trace

voc_labels = ('nine','ten','jack','queen','king','ace')

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

TEST_images_dir = 'json/TEST_images.json'
with open(os.path.join(TEST_images_dir), 'r') as j:
    TEST_images = json.load(j)

TEST_objects_dir = 'json/TEST_objects.json'
with open(os.path.join(TEST_objects_dir), 'r') as j:
    TEST_objects = json.load(j)

for image_id in range(len(TEST_images)):
    image_name = TEST_images[image_id].split('/')[-1].split('.')[0]
    print(image_name)
    with open("./input/ground-truth/"+image_name+".txt", "w") as new_f:
        for i in range(len(TEST_objects[image_id]['boxes'])):
            obj_name = voc_labels[TEST_objects[image_id]['labels'][i]-1]
            xmin = TEST_objects[image_id]['boxes'][i][0]
            ymin = TEST_objects[image_id]['boxes'][i][1]
            xmax = TEST_objects[image_id]['boxes'][i][2]
            ymax = TEST_objects[image_id]['boxes'][i][3]
            new_f.write("%s %s %s %s %s\n" % (obj_name, xmin, ymin, xmax, ymax))
print("Conversion completed!")
