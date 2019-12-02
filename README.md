# Caffe-HeatMap 

##
Caffe-Heatmap is modefied based on Caffe-SSD (https://github.com/weiliu89/caffe/tree/ssd)

### 
KeyPoint is used as bndbox the same way in SSD.
KeyPoint(x,y) is repeated in VOC bndbox:
xmin:x
ymin:y
xmax:x
ymax:y

####
The annotated_data_layer has new input param:
heatmap_c: output channel number
heatmap_h: output height
heatmap_w: output weight
heatmap_visual: for debug
heatmap_sigma: kernel R

#####
Some layers : data_transform, bbox_util, im_tansforms, caffeproto are also changed.

#####
Check models/slot-mbnv2-deep-down4.
I use mobilenetv2 as backbones to get key points.
![images](https://github.com/George-Tech/Misc/blob/master/Images/1_0411_1_000158_draw_map0.jpg)

Person Pose on FLIC.

![images](https://github.com/George-Tech/Misc/blob/master/Images/12-oclock-high-special-edition-00016481_draw_map0.jpg)
