# Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation (CVPR2019)
This is a [pytorch](http://pytorch.org/) implementation of [CLAN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf).

### Oral Presentation Video
[![Watch the video](https://github.com/RoyalVane/CLAN/blob/master/gifs/video.png)](https://www.bilibili.com/video/av53561336/)

### Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Pytorch 1.0.0

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The imagenet pretraind model]( https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2 )

The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   ├── SYNTHIA/ 
|   |   ├── RAND_CITYSCAPES/
│   └── 			
└── model/
│   ├── DeepLab_resnet_pretrained.pth
...
```

### Train
```
CUDA_VISIBLE_DEVICES=0 python CLAN_train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```

### Evaluate
```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate.py --restore-from  ./snapshots/GTA2Cityscapes/GTA5_100000.pth --save ./result/GTA2Cityscapes_100000
```
Our pretrained model is available via [Google Drive]( https://drive.google.com/open?id=1Hl7r6fIbNfyA9A8wGUJIMOwzXVQ61ik8 )

### Compute IoU
```
python CLAN_iou.py ./data/Cityscapes/gtFine/val result/GTA2Cityscapes_100000
```

#### Tip: The best-performance model might not be the final one in the last epoch. If you want to evaluate every saved models in bulk, please use CLAN_evaluate_bulk.py and CLAN_iou_bulk.py, the result will be saved in an Excel sheet.
```
CUDA_VISIBLE_DEVICES=0 python python CLAN_evaluate_bulk.py
python CLAN_iou_bulk.py
```

### Visualization Results
<p align="left">
	<img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_1.gif"  width="420" height="210" alt="(a)"/>

  <img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_2.gif"  width="420" height="210" alt="(b)"/>
</p>
<p align="left">
	<img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_3.gif"  width="420" height="210" alt="(c)"/>
  
  <img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_4.gif"  width="420" height="210" alt="(d)"/>
</p>

#### This code is heavily borrowed from the baseline [AdaptSegNet]( https://github.com/wasidennis/AdaptSegNet )

### Citation
If you use this code in your research please consider citing
```
@inproceedings{Yawei2019Taking,
title={Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation},
author={Luo, Yawei and Zheng, Liang and Guan, Tao and Yu, Junqing and Yang, Yi},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2019}
}
```
