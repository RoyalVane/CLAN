# Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation (CVPR2019)
A [pytorch](http://pytorch.org/) implementation of [CLAN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf).

### Oral Presentation Trailer
[![Watch the video](https://github.com/RoyalVane/CLAN/blob/master/gifs/video.png)](https://www.bilibili.com/video/av53561336/)

### Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Pytorch 1.0.0

### Getting started

Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )
Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )
Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )
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
└── 
```

### Train


### Visualization Results
<p align="left">
	<img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_1.gif"  width="420" height="210" alt="(a)"/>

  <img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_2.gif"  width="420" height="210" alt="(b)"/>
</p>
<p align="left">
	<img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_3.gif"  width="420" height="210" alt="(c)"/>
  
  <img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_4.gif"  width="420" height="210" alt="(d)"/>
</p>

## Citation
If you use this code in your research please consider citing
```
>@inproceedings{Yawei2019Taking,
title={Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation},
author={Luo, Yawei and Zheng, Liang and Guan, Tao and Yu, Junqing and Yang, Yi},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2019}
}
```
