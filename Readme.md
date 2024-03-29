# 3D-MOT

## Operating Environment

|    python     |                      3.7                       |
| :-----------: | :--------------------------------------------: |
|   **CUDA**    |                    **11.3**                    |
|    **OS**     |             **Ubuntu 20.04.1 LTS**             |
| **Processor** | **Intel® Core™ i9-10980XE CPU @ 3.00GHz × 36** |
| **Graphics**  |    **NVIDIA Corporation TU102 [TITAN RTX]**    |

## Data Preparation

```sh
# kitti数据集   
└── kitti
       ├── testing 
       |      ├──calib
       |      |    ├──0000.txt
       |      |    ├──....txt
       |      |    └──0028.txt
       |      ├──image_02
       |      |    ├──0000
       |      |    ├──....
       |      |    └──0028
       |      ├──pose
       |      |    ├──0000
       |      |    |    └──pose.txt
       |      |    ├──....
       |      |    └──0028
       |      |         └──pose.txt
       |      ├──label_02
       |      |    ├──0000.txt
       |      |    ├──....txt
       |      |    └──0028.txt
       |      └──velodyne
       |           ├──0000
       |           ├──....
       |           └──0028      
       └── training  # 与训练集结构一样
              ├──calib
              ├──image_02
              ├──pose
              ├──label_02
              └──velodyne 
└── casa
       ├── testing
       |      ├──0000
       |      ├──....
       |      └──0020
       ├── training
       |      ├──0000
       |      ├──....
       |      └──0020
```

## operating steps

1. configuration environment
   - `requirements.txt`
2. data preparation
   - https://www.cvlibs.net/datasets/kitti/eval_tracking.php
3. Follow the correct path to change `dataset_path`,`detections_path` in `config/online/casa_mot.yaml `
4. run `python3 kitti_3DMOT.py`
5. `result` can see the result

![125](./src/125.gif)
