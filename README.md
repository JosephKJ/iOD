## Incremental Object Detection via Meta-Learning
#### Published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 
##### DOI 10.1109/TPAMI.2021.3124133

Early access on IEEE Xplore: [https://ieeexplore.ieee.org/document/9599446](https://ieeexplore.ieee.org/document/9599446)

arXiv paper: [https://arxiv.org/abs/2003.08798](https://arxiv.org/abs/2003.08798)


<div align="center">
  <img src="https://user-images.githubusercontent.com/4231550/138396577-bdef2d95-5f00-47c4-bf90-927d7231f090.png"/>
</div>

## Abstract
In a real-world setting, object instances from new classes can be continuously encountered by object detectors. When existing object detectors are applied to such scenarios, their performance on old classes deteriorates significantly. A few efforts have been reported to address this limitation, all of which apply variants of knowledge distillation to avoid catastrophic forgetting. 

We note that although distillation helps to retain previous learning, it obstructs fast adaptability to new tasks, which is a critical requirement for incremental learning. In this pursuit, we propose a meta-learning approach that learns to reshape model gradients, such that information across incremental tasks is optimally shared. This ensures a seamless information transfer via a meta-learned gradient preconditioning that minimizes forgetting and maximizes knowledge transfer. In comparison to existing meta-learning methods, our approach is task-agnostic, allows incremental addition of new-classes and scales to high-capacity models for object detection. 

We evaluate our approach on a variety of incremental learning settings defined on PASCAL-VOC and MS COCO datasets, where our approach performs favourably well against state-of-the-art methods.

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/4231550/145962389-75511c27-3d9f-4dd2-be93-934dcdf4d70c.jpg" width="800" />
</p>

<p align="center" width="80%">
<strong>Figure:</strong> Qualitative results of our incremental object detector trained in a 10+10 setting where the first task contain instances of <i>aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair and cow</i>, while the second task learns instance from <i>diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train and tvmonitor</i>. Our model is able to detect instances of both tasks alike, without forgetting.
</p>


## Installation and setup
- Install the Detectron2 library that is packages along with this code base. See [INSTALL.md](INSTALL.md).
- Download and extract Pascal VOC 2007 to `./datasets/VOC2007/`
- Use the starter script: `run.sh`

## Trained Models and Logs

| Setting | Reported mAP | Reproduced mAP | Commands | Models and logs |
|:-------:|:------------:|:--------------:|:--------:|:---------------:|
|   19+1  |     70.2     |      70.4      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L1-L8)    |   [Google Drive](https://drive.google.com/file/d/1sW-aZ9crRFjgbErtgXNQ8hO67WLKYAAn/view?usp=sharing)  |
|   15+5  |     67.8     |      69.6      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L11-L19)  |   [Google Drive](https://drive.google.com/file/d/1E8m4VrrKmNYT1Zba0MwaI3ZjztrLobcA/view?usp=sharing)  |
|  10+10  |     66.3     |      67.3      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L22-L30)  |   [Google Drive](https://drive.google.com/file/d/1LH7OY-uMifl2gwCFEgm6U5h_Xfh1nPcH/view?usp=sharing)  |

##### Configurations with which the above results were reproduced:
- Python version: 3.6.7
- PyTorch version: 1.3.0
- CUDA version: 11.0
- GPUs: 4 x NVIDIA GTX 1080-ti

## Acknowledgement
The code is build on top of Detectron2 library. 


## Citation
If you find our research useful, please consider citing us:

```BibTeX
@ARTICLE {joseph2021incremental,
author = {Joseph. KJ and Jathushan. Rajasegaran and Salman. Khan and Fahad. Khan and Vineeth. N Balasubramanian},
journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
title = {Incremental Object Detection via Meta-Learning},
year = {2021},
issn = {1939-3539},
doi = {10.1109/TPAMI.2021.3124133},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {nov}
}

```
