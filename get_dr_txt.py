from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from ipdb import set_trace
import numpy as np
from utils import *
import torch
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect(original_image,img_name, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform

    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)
    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, det_boxes, det_labels, det_scores

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("/snap/gnome-3-28-1804/67/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i]+'    '+str(det_scores[0][i].item())[:5], fill='blue',     
                  font=font)
    del draw
    # annotated_image.save(result_dir + img_name) #保存
    # set_trace()
    det_scores = det_scores[0].to('cpu')
    return annotated_image, det_boxes, det_labels, det_scores

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[1.699449, 1.565750, 1.419592], std=[2.711962, 2.504607, 2.274658])

# 加载待检测列表
TEST_images_dir = 'json/TEST_images.json'
with open(os.path.join(TEST_images_dir), 'r') as j:
    TEST_images = json.load(j)

for images in TEST_images:
    image_name = images.split('/')[-1].split('.')[0]
    original_image = Image.open(images, mode='r')
    original_image = original_image.convert('RGB')
    original_image.save("./input/images-optional/"+image_name+".jpg")

    #detect(original_image, pic, min_score=0.05, max_overlap=0.5, top_k=200).show()
    detect_img, det_boxes, det_labels, det_scores = detect(original_image, image_name, min_score=0.05, max_overlap=0.05, top_k=200)
    # detect_img.show()
    if det_labels != ['background']:
        # 保存检测结果图像
        detect_img.save("./input/detection-results/"+image_name+".jpg")

        # 保存检测结果txt
        f = open("./input/detection-results/"+image_name+".txt","w") 
        for i in range(len(det_boxes)):
            f.write("%s %s %s %s %s %s\n" % (det_labels[i], str(det_scores[i].item())[:6], str(int(det_boxes[i][0].item())), str(int(det_boxes[i][1].item())), str(int(det_boxes[i][2].item())),str(int(det_boxes[i][3].item()))))
        f.close()
        print('{} finish!'.format(image_name))
    else:
        print('{} not detect image!'.format(image_name))
print("Conversion completed!")
    