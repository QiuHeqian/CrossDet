# CrossDet: Crossline Representation for Object Detection (ICCV 2021)
## Introduction  
Object detection aims to accurately locate and classify objects in an image, which requires precise object representations. Existing methods usually use rectangular anchor
boxes or a set of points to represent objects. However, these methods either introduce background noise or miss the continuous appearance information inside the object, and thus
cause incorrect detection results. In this paper, we propose a novel anchor-free object detection network, called CrossDet, which uses a set of growing cross lines along horizontal and vertical axes as object representations. An object can be ﬂexibly represented as cross lines in different combinations. It not only can effectively reduce the interference of noise, but also take into account the continuous object information, which is useful to enhance the discriminability of object features and fnd the object boundaries. Based on the learned cross lines, we propose a crossline extraction module to adaptively capture features of cross lines. Furthermore, we design a decoupled regression mechanism to regress the localization along the horizontal and vertical directions respectively, which helps to decrease the optimization diffculty because the optimization space is limited to a specifc direction. Our method achieves consistently improvement on the PASCAL VOC and MS-COCO datasets. The experiment results demonstrate the effectiveness of our proposed method.  
## Prerequisites
* MMDetection
* pytorch
* Please see [get_started.md](https://github.com/QiuHeqian/CrossDet/blob/master/docs/get_started.md) for installation and the basic usage of MMDetection.
## Train  

