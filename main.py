import os
import argparse
import warnings
from trainer import cross_domain_trainer
from trainer2 import cross_domain_trainer_ours
import sklearn.exceptions
import pickle
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='EEG-TFAC',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, ')
parser.add_argument('--run_description',        default='EEG-TFAC',                     type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='TFAC',               type=str, help='DANN, Deep_Coral, TFAC, MMDA, VADA, DIRT, CDAN, AdaMatch, HoMM, CoDATS')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='EEG',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA, Boiler)')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')

args = parser.parse_args()

if __name__ == "__main__":
    if args.da_method =='TFAC':
        trainer = cross_domain_trainer_ours(args)
        trainer.train()
    else:
        trainer = cross_domain_trainer(args)
        trainer.train()
        # trainer.visualize()
        # dic = {'1':trainer.src_all_features,'2':trainer.src_true_labels,'3':trainer.trg_all_features,'4':trainer.trg_true_labels,'acc': trainer.trg_acc_list}
        # with open('saved_dictionary2.pickle', 'wb') as handle:
        #     pickle.dump(dic, handle)
       