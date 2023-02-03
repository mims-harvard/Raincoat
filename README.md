
## Requirmenets:
- Python3
- Pytorch==1.7
- Numpy==1.20.1
- scikit-learn==0.24.1
- Pandas==1.2.4
- skorch==0.10.0 
- openpyxl==3.0.7 
- Wandb=0.12.7 

## Datasets
### Loading and Preparing Benchmark Datasets
Create a folder and download the pre-processed versions of the datasets [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B), [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ), [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO), [Boiler](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO), and [Sleep-EDF](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO).

To add new dataset (*e.g.,* NewData), it should be placed in a folder named: NewData in the datasets directory.

Since "NewData" has several domains, each domain should be split into train/test splits with naming style as
"train_*x*.pt" and "test_*x*.pt".

The structure of data files should in dictionary form as follows:
`train.pt = {"samples": data, "labels: labels}`, and similarly for `test.pt`.

#### Configurations
Next, you have to add a class with the name NewData in the `configs/data_model_configs.py` file. 
You can find similar classes for existing datasets as guidelines. 
Also, you have to specify the cross-domain scenarios in `self.scenarios` variable.

Last, you have to add another class with the name NewData in the `configs/hparams.py` file to specify
the training parameters.


## Closed-Set Domain Adaptation Algorithms
### Baselines
- [Deep Coral](https://arxiv.org/abs/1607.01719)
- [CDAN](https://arxiv.org/abs/1705.10667)
- [DIRT-T](https://arxiv.org/abs/1802.08735)
- [HoMM](https://arxiv.org/pdf/1912.11976.pdf)
- [CoDATS](https://arxiv.org/pdf/2005.10996.pdf)
- [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf)
- [CLUDA](https://www.ijcai.org/proceedings/2021/0378.pdf)

## Universal Domain Adaptation Algorithms
### Existing Algorithms
- [UniDA](https://arxiv.org/abs/1607.01719)
- [DANCE](https://arxiv.org/abs/1705.10667)
- [OVANet](https://arxiv.org/abs/1705.10667)
- [UniOT](https://arxiv.org/abs/1802.08735)


## RAINCOAT
### Model
Our main model architecture can be found [here](algorithms/RAINCOAT.py). 

### Algorithm 

## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.
- For example, if we want to experiment different UDA methods with CNN backbone, we can assign
`--experiment_description CNN_backnones --run_description DANN` and `--experiment_description CNN_backnones --run_description DDC` and so on.

### Training a model

To train a model:

```
python main.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method DANN \
                --dataset HHAR \
                --backbone CNN \
                --num_runs 5 \
```
