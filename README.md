本代码在原作者的基础上增加了部分功能，主要用于个人学习ssd流程：
新增部分：
（１）原程序无法在断网的情况下进行初始化网络，原因在于加载vgg16预训练模型的时候从torchvision.models.vgg16加载，需要联网，本程序将model.py中的这三行代码
              # Pretrained VGG base
              # pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
              # pretrained_param_names = list(pretrained_state_dict.keys())
      更改为：
              vgg16_model_path = 'pretrained/vgg16-397923af.pth'
              pretrained_state_dict = torch.load(vgg16_model_path)
              pretrained_param_names = list(pretrained_state_dict.keys()
      这三行。预训练vgg16权重存放在pretrained文件夹中，并给出百度云地址。
 
（２）原程序中产生8732个候选框的设置中，fmap_dims、obj_scales，aspect_ratios，这三个变量采用普通字典形式，但是python更新后，普通字典在索引时是乱序的，导致每次训练产生的候选框不同，导致训练出现问题，远远达不到预期效果，将这三个变量更换为OrderedDict类型，在逐个索引时有顺序，解决了上述问题，具体代码看model.py中的create_prior_boxes函数。

（３）出于本人新手学习考虑，在程序中增加了４个可视化内容，程序位于utils.py底部：
      １）show_groundtruth_priorbox_predictbox－－－－－－在原图中可视化真值框，初始框，预测框－－－－－－－引用位置：model.py
      ２）show_feature_map －－－－－－－－－－－－－－－可视化提取的特征图－－－－－－－－－－－－－－－－－引用位置：model.py
      ３）show_train_pic－－－－－－－－－－－－－－－－－可视化放入模型训练图像（经过数据增强后的图像）－－－引用位置：train.py
      ４）show_transform_pic－－－－－－－－－－－－－－－可视化数据增强过程中图像的变化－－－－－－－－－－－引用位置：utils.py

（４）原程序在训练时无法测试，本程序增加了测试程序，位于train.py
        # eval model
        if epoch % eval_freq == 0:    
            evaluate(test_loader, model, epoch)
并生成AP.txt和log.txt，将生成的这两个结果放入draw文件夹中，运行draw_loss_precision.py会生成loss图和precision图。（注意：需要根据检测类别的不同简单修改draw_loss_precision.py中的内容）

（５）原程序能够计算并显示训练好的模型的map，但是感觉结果普遍较高，但是根据检测的效果来看，并不理想。参考了另一个博主的map计算方式，并迁移到本程序中。
      １）get_dr_txt.py－－－－－－－－－－－获得检测结果txt
      ２）get_gt_txt.py－－－－－－－－－－－获得真值txt
      ３）get_map_txt.py－－－－－－－－－－根据以上两个txt计算模型的map
      ４）intersect-get-and-dr.py－－－－－－如果检测结果的txt数量和真值txt数量不同，用这个程序使其相同。
      相关博客参考：https://blog.csdn.net/weixin_44791964/article/details/104695264
     
（６）原始测试图片的程序只能测试一张，增加测试多张图像，原始图像存放在test_data，检测结果存放在test_val_result，所用程序为detect.py。

（７）增加初始框可视化（直接运行prior_boxes_show中的Vision_for_prior.py文件）
问题：
由于python版本不同，可能会出现一些问题。
model.py中suppress = torch.max(suppress, overlap[box] > max_overlap)这句话可能会出现
RuntimeError: Expected object of scalar type Byte but got scalar type Bool for argument #2 'other' in call to _th_max这个问题，适当修改数据类型即可解决

原作者程序参考：原作者README.md
