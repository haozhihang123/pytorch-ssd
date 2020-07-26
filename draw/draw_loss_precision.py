# -*- coding:UTF8 -*-
import re
import matplotlib.pyplot as plt 
from ipdb import set_trace
import random
def draw_loss():
    epoch = []
    loss = []
    log_file_path = "log.txt"
    for line in open(log_file_path):   
        result1 = int(re.findall('Epoch: \[(.*?)\]',line)[0])
        epoch.append(result1)
        result2 = float(re.findall('\((.*?)\)',line)[0])
        loss.append(result2)
    set_trace()
    plt.plot(epoch,loss,label='iterations-loss',linewidth=3,color='r',markerfacecolor='blue',markersize=12) 
    plt.xlabel('iterations')
    plt.ylabel('loss') 
    plt.title('iterations-loss')
    plt.legend()
    plt.savefig('./iterations-loss.jpg')
    plt.close()

def draw_precision(labels):
    # 设置变量
    iterations = []
    precision_label = {}
    for label in labels:
        precision_label[label] = []
    # 获取数据
    ap_file_path = 'AP.txt'
    for line in open(ap_file_path):   
        data = re.findall(' : \[(.*?)\]',line)
        set_trace()
        for i,data_ in enumerate(data):
            precision_label[labels[i]].append(float(data_))
    iterations = [i for i in range(len(precision_label[labels[0]]))]
    # 开始绘图,每一类都画一个，最终画map
    for i in range(len(labels)):
        plt.plot(iterations,precision_label[labels[i]],label=labels[i],linewidth=3,color=(random.random(),random.random(),random.random() ),markerfacecolor='blue',markersize=12) 
        plt.xlabel('iterations')
        plt.ylabel('precision') 
        plt.title('iterations-precision')
        plt.xlim(0, 18)
        # plt.ylim(0,18)
        plt.legend(loc = 'upper right')
        plt.savefig('./iterations-precision.jpg')
    plt.close() 



if __name__ == '__main__':
    # draw_loss()
    labels = ('nine','ten','jack','queen','king','ace','map')
    draw_precision(labels)
