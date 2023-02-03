#!/bin/sh

conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-CDAN --run_description HHAR_SA-CDAN --da_method CDAN --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-CoDATS --run_description HHAR_SA-CoDATS --da_method CoDATS --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-AdvSKM --run_description HHAR_SA-AdvSKM --da_method AdvSKM --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-HoMM --run_description HHAR_SA-HoMM --da_method HoMM --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-Deep_Coral --run_description HHAR_SA-Deep_Coral --da_method Deep_Coral --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-AdaMatch --run_description HHAR_SA-AdaMatch --da_method AdaMatch --dataset HHAR_SA
conda run -n cuda116 python -u main.py --experiment_description HHAR_SA-DIRT --run_description HHAR_SA-DIRT --da_method DIRT --dataset HHAR_SA
