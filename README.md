# P2PNet (ICCV2021 Oral Presentation)

This repository contains codes for the official implementation in PyTorch of **P2PNet** as described in [Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework](https://arxiv.org/abs/2107.12746).
 
A brief introduction of P2PNet can be found at [机器之心 (almosthuman)](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650827826&idx=3&sn=edd3d66444130fb34a59d08fab618a9e&chksm=84e5a84cb392215a005a3b3424f20a9d24dc525dcd933960035bf4b6aa740191b5ecb2b7b161&mpshare=1&scene=1&srcid=1004YEOC7HC9daYRYeUio7Xn&sharer_sharetime=1633675738338&sharer_shareid=7d375dccd3b2f9eec5f8b27ee7c04883&version=3.1.16.5505&platform=win#rd).

The codes is tested with PyTorch 1.5.0. It may not run with other versions.

## Visualized demos for P2PNet
<img src="vis/congested1.png" width="1000"/>   
<img src="vis/congested2.png" width="1000"/> 
<img src="vis/congested3.png" width="1000"/> 

## The network
The overall architecture of the P2PNet. Built upon the VGG16, it firstly introduce an upsampling path to obtain fine-grained feature map. 
Then it exploits two branches to simultaneously predict a set of point proposals and their confidence scores.

<img src="vis/net.png" width="1000"/>   

## Comparison with state-of-the-art methods
The P2PNet achieved state-of-the-art performance on several challenging datasets with various densities.

| Methods   | Venue     | SHTechPartA <br> MAE/MSE  |SHTechPartB <br> MAE/MSE | UCF_CC_50 <br> MAE/MSE | UCF_QNRF <br> MAE/MSE   |
|:----:|:----:|:----:|:----:|:----:|:----:|
CAN  | CVPR'19 | 62.3/100.0 | 7.8/12.2 | 212.2/**243.7** | 107.0/183.0 |
Bayesian+ | ICCV'19 | 62.8/101.8 | 7.7/12.7 | 229.3/308.2 | 88.7/154.8 |
S-DCNet  | ICCV'19 | 58.3/95.0 | 6.7/10.7 | 204.2/301.3 | 104.4/176.1 |
SANet+SPANet  | ICCV'19 | 59.4/92.5 | 6.5/**9.9** | 232.6/311.7 | -/- |
DUBNet  | AAAI'20 | 64.6/106.8 | 7.7/12.5 | 243.8/329.3 | 105.6/180.5 |
SDANet | AAAI'20 | 63.6/101.8 | 7.8/10.2 | 227.6/316.4 | -/- |
ADSCNet | CVPR'20 | <u>55.4</u>/97.7 | <u>6.4</u>/11.3 | 198.4/267.3 | **71.3**/**132.5**|
ASNet   | CVPR'20 | 57.78/<u>90.13</u> | -/- | <u>174.84</u>/<u>251.63</u> | 91.59/159.71 |
AMRNet  | ECCV'20 | 61.59/98.36 | 7.02/11.00 | 184.0/265.8 | 86.6/152.2 |
AMSNet  | ECCV'20 | 56.7/93.4 | 6.7/10.2 | 208.4/297.3 | 101.8/163.2|
DM-Count  | NeurIPS'20 | 59.7/95.7 | 7.4/11.8 | 211.0/291.5 | 85.6/<u>148.3</u>|
**Ours** |- | **52.74**/**85.06** | **6.25**/**9.9** | **172.72**/256.18 | <u>85.32</u>/154.5 |

Comparison on the [NWPU-Crowd](https://www.crowdbenchmark.com/resultdetail.html?rid=81) dataset.

| Methods   | MAE[O]  |MSE[O] | MAE[L] | MAE[S]   |
|:----:|:----:|:----:|:----:|:----:|
MCNN  | 232.5|714.6 | 220.9|1171.9 |
SANet  | 190.6 | 491.4 | 153.8 | 716.3|
CSRNet | 121.3 | 387.8 | 112.0 | <u>522.7</u> |
PCC-Net  | 112.3 | 457.0 | 111.0 | 777.6 |
CANNet  | 110.0 | 495.3 | 102.3 | 718.3|
Bayesian+  | 105.4 | 454.2 | 115.8 | 750.5 |
S-DCNet   | 90.2 | 370.5 | **82.9** | 567.8 |
DM-Count  | <u>88.4</u> | 388.6 | 88.0 | **498.0** |
**Ours** | **77.44**|**362** | <u>83.28</u>| 553.92 |

The overall performance for both counting and localization.

|nAP$_{\delta}$|SHTechPartA| SHTechPartB | UCF_CC_50 | UCF_QNRF | NWPU_Crowd |
|:----:|:----:|:----:|:----:|:----:|:----:|    
$\delta=0.05$ | 10.9\% | 23.8\%  | 5.0\% | 5.9\% | 12.9\% | 
$\delta=0.25$ | 70.3\% | 84.2\%  | 54.5\% | 55.4\% | 71.3\% |  
$\delta=0.50$ | 90.1\% | 94.1\%  | 88.1\% | 83.2\% | 89.1\% | 
$\delta=\{{0.05:0.05:0.50}\}$ | 64.4\% | 76.3\%  | 54.3\% | 53.1\% | 65.0\% |  

Comparison for the localization performance in terms of F1-Measure on NWPU.

| Method| F1-Measure |Precision| Recall |
|:----:|:----:|:----:|:----:|
FasterRCNN  |  0.068 |  0.958 | 0.035 |
TinyFaces |  0.567  |  0.529 | 0.611 |
RAZ |   0.599 |  0.666 |  0.543|
Crowd-SDNet |  0.637  | 0.651  | 0.624  |
PDRNet |  0.653 | 0.675  | 0.633  |
TopoCount | 0.692  | 0.683  | **0.701** |
D2CNet | <u>0.700</u> | **0.741**  | 0.662 |
**Ours** |**0.712** | <u>0.729</u>  | <u>0.695</u> |

## Installation
* Clone this repo into a directory named P2PNET_ROOT
* Organize your datasets as required
* Install Python dependencies. We use python 3.6.5 and pytorch 1.5.0
```
pip install -r requirements.txt
```

## Organize the counting dataset
We use a list file to collect all the images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
train/scene02/img01.jpg train/scene02/img01.txt
```

### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
```
DATA_ROOT is your path containing the counting datasets.

### Annotations format
For the annotations of each image, we use a single txt file which contains one annotation per line. Note that indexing for pixel values starts at 0. The expected format of each line is:
```
x1 y1
x2 y2
...
```

## Training

The network can be trained using the `train.py` script. For training on SHTechPartA, use

```
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

A trained model (with an MAE of **51.96**) on SHTechPartA is available at "./weights", run the following commands to launch a visualization demo:

```
CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/
```

## Acknowledgements

- Part of codes are borrowed from the [C^3 Framework](https://github.com/gjy3035/C-3-Framework).
- We refer to [DETR](https://github.com/facebookresearch/detr) to implement our matching strategy.


## Citing P2PNet

If you find P2PNet is useful in your project, please consider citing us:

```BibTeX
@inproceedings{song2021rethinking,
  title={Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework},
  author={Song, Qingyu and Wang, Changan and Jiang, Zhengkai and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Wu, Yang},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Related works from Tencent Youtu Lab
- [AAAI2021] To Choose or to Fuse? Scale Selection for Crowd Counting. ([paper link](https://ojs.aaai.org/index.php/AAAI/article/view/16360) & [codes](https://github.com/TencentYoutuResearch/CrowdCounting-SASNet))
- [ICCV2021] Uniformity in Heterogeneity: Diving Deep into Count Interval Partition for Crowd Counting. ([paper link](https://arxiv.org/abs/2107.12619) & [codes](https://github.com/TencentYoutuResearch/CrowdCounting-UEPNet))
