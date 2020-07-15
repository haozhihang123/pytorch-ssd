# pytorch-ssd
上传缘由：
源程序来自于：https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection。
但是原作者的环境是： `PyTorch 0.4` in `Python 3.6`。在用我的环境运行代码时发生一系列问题，最终费了不少力气解决。
并对原代码做了一些扩充：打印log并生成图像，批量检测数据，目前只是目标检测小白，以此记录自己的学习生涯并备份程序。

使用过程：
１．训练
这个ssd原项目用的是VOC2007和VOC2012.经过本人修改后可以实现对自己的数据集进行训练。具体操作是在creat_data_list.py中更改自己的VOC数据集目录。例如：voc07_path='/home/haozhihang/data/VOCdevkit/VOC2007'
最后的/VOCdevkit/VOC2007是官方VOC数据格式。具体的训练超参数设置在train.py（不设置也可以跑）。然后在pytorch环境中运行即可。训练生成的模型文件保存在根部录下。

２．测试
下载项目后不可以直接运行：运行此程序需要在根目录下有一个训练生成成的checkpoint_ssd300.pth.tar模型文件
detect.py是用来测试图片并生成相应的带框图片：待测试图片放在test_data文件夹中，输出在test_val_result。
eval.py 是计算MAP并将结果保存在test_val_result．

３．可视化
增加四项可视化内容：
（１）可视化特征图并保存
定义：utils.py→show_feature_map
引用：model.py→376
状态：一般训练时处于注释状态，当需要可视化特征图时打开，特征图全部保存后会有中断
（２）可视化样本训练时真值框，初始框（与真值框IOU最大的一个），预测框并保存
定义：utils.py→show_groundtruth_priorbox_predictbox
引用：model.py→604
状态：一般训练时处于注释状态，当需要可视化时时打开，全部保存后会有中断
（３）可视化用于训练的样本并保存
定义：utils.py→show_train_pic
引用：train.py→128
状态：一般训练时处于注释状态，当需要可视化时时打开，全部保存后会有中断
（4）可视化数据增强各个阶段的图片并保存
定义：utils.py→show_transform_pic
引用：tuils.py→671
状态：一般训练时处于注释状态，当需要可视化时时打开，全部保存后会没有中断，需要在train.py中的126行加入中断，或者手动中断

４．具体程序内容看原作者README.md
