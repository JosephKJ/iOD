## Incremental Object Detection via Meta-Learning
#### To appear in an upcoming issue of the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

arXiv paper: [https://arxiv.org/abs/2003.08798](https://arxiv.org/abs/2003.08798)

<div align="center">
  <img src="https://user-images.githubusercontent.com/4231550/138396577-bdef2d95-5f00-47c4-bf90-927d7231f090.png"/>
</div>

## Abstract
In a real-world setting, object instances from new classes can be continuously encountered by object detectors. When existing object detectors are applied to such scenarios, their performance on old classes deteriorates significantly. A few efforts have been reported to address this limitation, all of which apply variants of knowledge distillation to avoid catastrophic forgetting. 

We note that although distillation helps to retain previous learning, it obstructs fast adaptability to new tasks, which is a critical requirement for incremental learning. In this pursuit, we propose a meta-learning approach that learns to reshape model gradients, such that information across incremental tasks is optimally shared. This ensures a seamless information transfer via a meta-learned gradient preconditioning that minimizes forgetting and maximizes knowledge transfer. In comparison to existing meta-learning methods, our approach is task-agnostic, allows incremental addition of new-classes and scales to high-capacity models for object detection. 

We evaluate our approach on a variety of incremental learning settings defined on PASCAL-VOC and MS COCO datasets, where our approach performs favourably well against state-of-the-art methods.

## Installation and setup
- Install the Detectron2 library that is packages along with this code base. See [INSTALL.md](INSTALL.md).
- Download and extract Pascal VOC 2007 to `./datasets/VOC2007/`
- Use the starter script: `run.sh`

## Trained Models and Logs

| Setting | Reported mAP | Reproduced mAP | Commands | Models and logs |
|:-------:|:------------:|:--------------:|:--------:|:---------------:|
|   19+1  |     70.2     |      70.4      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L1-L8)    |   [Google Drive](https://drive.google.com/file/d/1pocjYPenjXda0fRh7ir_c1ItyAZCBoEN/view?usp=sharing)  |
|   15+5  |     67.8     |      69.6      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L11-L19)  |   [Google Drive](https://drive.google.com/file/d/1KaynMWxb6nHytfMYP_wh8Dy-AvsLLazQ/view?usp=sharing)  |
|  10+10  |     66.3     |      67.3      |  [run.sh](https://github.com/JosephKJ/iOD/blob/main/run.sh#L22-L30)  |   [Google Drive](https://drive.google.com/file/d/1aWc-1P7ZtNrye_asN5mKMtu7G8G0tLAm/view?usp=sharing)  |

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
@article{joseph2021incremental,
  title={Incremental object detection via meta-learning},
  author={Joseph, KJ and Rajasegaran, Jathushan and Khan, Salman and Khan, Fahad Shahbaz and Balasubramanian, Vineeth},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
