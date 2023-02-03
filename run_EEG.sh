#!/bin/sh

conda run -n cuda116 python -u main.py --experiment_description EEG-CDAN --run_description EEG-CDAN --da_method CDAN --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-CoDATS --run_description EEG-CoDATS --da_method CoDATS --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-AdvSKM --run_description EEG-AdvSKM --da_method AdvSKM --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-HoMM --run_description EEG-HoMM --da_method HoMM --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-Deep_Coral --run_description EEG-Deep_Coral --da_method Deep_Coral --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-AdaMatch --run_description EEG-AdaMatch --da_method AdaMatch --dataset EEG
conda run -n cuda116 python -u main.py --experiment_description EEG-DIRT --run_description EEG-DIRT --da_method DIRT --dataset EEG
