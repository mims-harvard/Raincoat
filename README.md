## Authors: [Huan He](https://hehuannb.github.io/) (hehuannb@gmail.com), [Owen Queen](https://www.linkedin.com/in/owen-queen-0348561a2/), [Teddy Koker](https://teddykoker.com/), [Consuelo Cuevas](https://consuelo-cuevas.editorx.io/home), [Theodoros Tsiligkaridis](https://github.com/mims-harvard/Raindrop)(ttsili@ll.mit.edu), [Marinka Zitnik](https://zitniklab.hms.harvard.edu/) (marinka@hms.harvard.edu)

## [Project website](https://zitniklab.hms.harvard.edu/projects/Raincoat/)

## Raincoat Paper: [ICML 2023](https://arxiv.org/abs/2302.03133)

## Overview of Raincoat

The transfer of models trained on labeled datasets from a source domain to unlabeled target domains is facilitated by unsupervised domain adaptation (UDA). However, when dealing with complex time series models, transferability becomes challenging due to differences in dynamic temporal structures between domains, which can result in feature shifts and gaps in time and frequency representations. Additionally, the label distributions in the source and target domains can be vastly different, making it difficult for UDA to address label shifts and recognize labels unique to the target domain. Raincoat is a domain adaptation method for time series that can handle both feature and label shifts.

<p align="center">
<img src="https://zitniklab.hms.harvard.edu/img/Raincoat-method.png">
</p>

## Installation and Setup

### 1: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/mims-harvard/Raincoat
cd Raincoat
```

### 2: Set Up Environment

To install the core environment dependencies of Raincoat, use requirement.txt:
```
pip: -r requirements.txt
```

    - Python3
    - Pytorch==1.7
    - Numpy==1.20.1
    - scikit-learn==0.24.1
    - Pandas==1.2.4
    - skorch==0.10.0 
    - openpyxl==3.0.7 
    
### 3: Download Datasets
Create a folder and download the pre-processed versions of the datasets [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B), [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ), [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO), [Boiler](https://researchdata.https://github.com/DMIRLAB-Group/SASA/tree/main/datasets/Boiler), and [Sleep-EDF](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9).

To add new dataset (*e.g.,* NewData), it should be placed in a folder named: NewData in the datasets directory.

Since "NewData" has several domains, each domain should be split into train/test splits with naming style as
"train_*x*.pt" and "test_*x*.pt".

The structure of data files should in dictionary form as follows:
`train.pt = {"samples": data, "labels: labels}`, and similarly for `test.pt`.



### 4: Set up configurations
Next, add a class with the name NewData in the `configs/data_model_configs.py` file. 
You can find similar classes for existing datasets as guidelines. 
Also, you have to specify the cross-domain scenarios in `self.scenarios` variable.
Last, you have to add another class with the name NewData in the `configs/hparams.py` file to specify
the training parameters.

<!-- ### Closed-Set Domain Adaptation Algorithms
#### Baselines
- [Deep Coral](https://arxiv.org/abs/1607.01719)
- [CDAN](https://arxiv.org/abs/1705.10667)
- [DIRT-T](https://arxiv.org/abs/1802.08735)
- [HoMM](https://arxiv.org/pdf/1912.11976.pdf)
- [CoDATS](https://arxiv.org/pdf/2005.10996.pdf)
- [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf)
- [CLUDA](https://openreview.net/forum?id=xPkJYRsQGM)

### Universal Domain Adaptation Algorithms
#### Existing Algorithms
- [UniDA](https://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf)
- [DANCE](https://cs-people.bu.edu/keisaito/research/DANCE.html)
- [OVANet](https://arxiv.org/abs/2104.03344)
- [UniOT](https://arxiv.org/abs/2210.17067) -->


## Usage

### Example
Please check out the [jupyter notebook](example.ipynb) for understanding RAINCOAT. 

### Model
Our main model architecture can be found [here](models/models.py). The freuqency encoder is defined in  

### Algorithm 
The algorithm can be found [here](algorithms/RAINCOAT.py). 
The implementation is build upon a published benchmark work [Adatime](https://arxiv.org/abs/2203.08321). 


### Training a Model

The experiments are organised in a hierarchical way such that:
- Experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.
- Each experiment could have different independent runs, which is determined by `--num_runs`.
- For example, if we want to experiment different UDA methods using Raincoat on WISDM dataset , we can assign
`--da_method RAINCOAT --run_description DANN` and `--dataset WISDM --num_runs 5` and so on.

To train a model:

```
python main.py  --experiment_description WISDM  \
                --run_description W_RAINCOAT \
                --da_method RAINCOAT \
                --dataset WISDM \
                --backbone CNN \
                --num_runs 5 \
```


An example bash script is provided in `scripts/run_WISDM.sh`.

To see and/or modify the default hyperparameters, please see `configs/hparams.py` and `configs/data_model_configs.py`.

### Citation
If you find *Raincoat* useful for your research, please consider citing this paper:

```
@inproceedings{he2023domain,
title = {Domain Adaptation for Time Series Under Feature and Label Shifts},
author = {He, Huan and Queen, Owen and Koker, Teddy and Cuevas, Consuelo and Tsiligkaridis, Theodoros and Zitnik, Marinka},
booktitle = {International Conference on Machine Learning},
year      = {2023}
}
```

### Lisence
Raincoat codebase is under MIT license. For individual dataset usage, please refer to the dataset license found in the website.