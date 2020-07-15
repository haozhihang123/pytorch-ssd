from utils import create_data_lists
from utils import create_voc_only_data_lists

if __name__ == '__main__':
    create_voc_only_data_lists(voc07_path='/home/haozhihang/data/VOCdevkit/all_data/puker', output_folder='./json')
#    create_data_lists(voc07_path='/home/haozhihang/haozhihang/pytorch/ssd/ssd.pytorch-master/data/data/VOCdevkit/VOC2007',
#                      #voc12_path='/home/haozhihang/data/VOCdevkit/VOC2012',
 #                     output_folder='./')
