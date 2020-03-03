# -*- coding:UTF8 -*-
import re
import matplotlib.pyplot as plt 

epoch = []
loss = []
for line in open("log.txt"):   
    result1 = int(re.findall('Epoch: \[(.*?)\]',line)[0])
    epoch.append(result1)
    result2 = float(re.findall('\((.*?)\)',line)[0])
    loss.append(result2)
plt.plot(epoch,loss,label='epoch-loss',linewidth=3,color='r',
markerfacecolor='blue',markersize=12) 
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.title('epoch-loss')
plt.legend()
plt.savefig('./epoch-loss.jpg')
plt.show() 