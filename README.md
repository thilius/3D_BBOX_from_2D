# 3D_BBOX_from_2D
Estimate the 3D bounding box from the 2D bounding box，which was detected by the YOLOv4 detector. The training dataset was made by BOXY dataset and KITTI dataset, in order to generate the 3D bounding box in car view.

Rely on:
+ `Pytorch`
+ `Argparse`
+ `numpy`
+ `OpenCV`

Usage:
+ detect with `bbox_3d_prediction.py`
+ train with `train.py`
+ make the dataset with the `kitti2dataset.py` or `boxy2dataset.py`
</br>
Result:
</br>
<img src=https://github.com/thilius/3D_BBOX_from_2D/blob/main/KITTI_Dataset/result/1_res.jpg width=60% />
<img src=https://github.com/thilius/3D_BBOX_from_2D/blob/main/KITTI_Dataset/result/1478212844.679978_res.jpg width=60% />
<img src=https://github.com/thilius/3D_BBOX_from_2D/blob/main/KITTI_Dataset/result/000451_res.jpg width=60% />
