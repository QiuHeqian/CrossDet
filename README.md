# [CrossDet: Crossline Representation for Object Detection (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiu_CrossDet_Crossline_Representation_for_Object_Detection_ICCV_2021_paper.pdf) 
## Introduction  
Object detection aims to accurately locate and classify objects in an image, which requires precise object representations. Existing methods usually use rectangular anchor
boxes or a set of points to represent objects. However, these methods either introduce background noise or miss the continuous appearance information inside the object, and thus cause incorrect detection results. In this paper, we propose a novel anchor-free object detection network, called CrossDet, which uses a set of growing cross lines along horizontal and vertical axes as object representations. An object can be ﬂexibly represented as cross lines in different combinations. It not only can effectively reduce the interference of noise, but also take into account the continuous object information, which is useful to enhance the discriminability of object features and fnd the object boundaries. Based on the learned cross lines, we propose a crossline extraction module to adaptively capture features of cross lines. Furthermore, we design a decoupled regression mechanism to regress the localization along the horizontal and vertical directions respectively, which helps to decrease the optimization diffculty because the optimization space is limited to a specifc direction. Our method achieves consistently improvement on the PASCAL VOC and MS-COCO datasets. The experiment results demonstrate the effectiveness of our proposed method.  
## Installation
* MMDetection
* pytorch
* Please see [get_started.md](https://github.com/QiuHeqian/CrossDet/blob/master/docs/get_started.md) for installation and the basic usage of MMDetection.
## Train  
```
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'.
./tools/dist_train.sh configs/crossdet/crossdet_r50_fpn_1x_coco.py 8
```
```
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with VOC dataset in 'data/VOCdevkit/'.
./tools/dist_train.sh configs/crossdet/crossdet_r50_fpn_1x_voc.py 8
```

## Inference
```
./tools/dist_test.sh configs/crossdet/crossdet_r50_fpn_1x_coco.py work_dirs/crossdet_r50_fpn_1x_coco/epoch_12.pth 8 --eval bbox
```
## Acknowledgement
Thanks MMDetection team for the wonderful open source project!

## Citition
If you use this code in your research, please cite this project.  

@inproceedings{qiu2021crossdet,  
  title={CrossDet: Crossline Representation for Object Detection},  
  author={Qiu, Heqian and Li, Hongliang and Wu, Qingbo and Cui, Jianhua and Song, Zichen and Wang, Lanxiao and Zhang, Minjian},  
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},  
  pages={3195--3204},  
  year={2021}  
}  

