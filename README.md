### Incremental Object Detection via Meta-Learning
To appear in an upcoming issue of the IEEE Transactions on Pattern Analysis and Machine Intelligence (*TPAMI*)

<div align="center">
  Teaser Figure
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### Abstract
In a real-world setting, object instances from new classes can be continuously encountered by object detectors. When existing object detectors are applied to such scenarios, their performance on old classes deteriorates significantly. A few efforts have been reported to address this limitation, all of which apply variants of knowledge distillation to avoid catastrophic forgetting. 

We note that although distillation helps to retain previous learning, it obstructs fast adaptability to new tasks, which is a critical requirement for incremental learning. In this pursuit, we propose a meta-learning approach that learns to reshape model gradients, such that information across incremental tasks is optimally shared. This ensures a seamless information transfer via a meta-learned gradient preconditioning that minimizes forgetting and maximizes knowledge transfer. In comparison to existing meta-learning methods, our approach is task-agnostic, allows incremental addition of new-classes and scales to high-capacity models for object detection. 

We evaluate our approach on a variety of incremental learning settings defined on PASCAL-VOC and MS COCO datasets, where our approach performs favourably well against state-of-the-art methods.

<div align="center">
  Architecture Figure
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>


### Installation and setup
- Install the Detectron2 library that is packages along with this code base. See [INSTALL.md](INSTALL.md).
- Download and extract Pascal VOC 2007 to `./datasets/VOC2007/`
- Use the starter script: `run.sh`

### Trained Models and Logs

| Setting | Reported mAP | Reproduced mAP | Commands | Models and logs |
|:-------:|:------------:|:--------------:|:--------:|:---------------:|
|   19+1  |     70.2     |      70.3      |  run.sh  |   Google Drive  |
|   15+5  |     67.8     |      69.6      |  run.sh  |   Google Drive  |
|  10+10  |     66.3     |      65.6      |  run.sh  |   Google Drive  |


### Acknowledgement
The code is build on top of Detectron2 library. 


### Citation
If you find our research useful, please cite us:

```BibTeX
@article{joseph2021incremental,
  title={Incremental object detection via meta-learning},
  author={Joseph, KJ and Rajasegaran, Jathushan and Khan, Salman and Khan, Fahad Shahbaz and Balasubramanian, Vineeth},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
