#!/bin/sh

conda run -n cuda116 python -u main.py --experiment_description HAR-CDAN --run_description HAR-CDAN --da_method CDAN --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-CoDATS --run_description HAR-CoDATS --da_method CoDATS --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-AdvSKM --run_description HAR-AdvSKM --da_method AdvSKM --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-HoMM --run_description HAR-HoMM --da_method HoMM --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-Deep_Coral --run_description HAR-Deep_Coral --da_method Deep_Coral --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-AdaMatch --run_description HAR-AdaMatch --da_method AdaMatch --dataset HAR
conda run -n cuda116 python -u main.py --experiment_description HAR-DIRT --run_description HAR-DIRT --da_method DIRT --dataset HAR
