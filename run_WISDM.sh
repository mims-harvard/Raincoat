conda run -n cuda116 python -u main.py --experiment_description WISDM-CDAN --run_description WISDM-CDAN --da_method CDAN --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-CoDATS --run_description WISDM-CoDATS --da_method CoDATS --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-AdvSKM --run_description WISDM-AdvSKM --da_method AdvSKM --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-HoMM --run_description WISDM-HoMM --da_method HoMM --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-Deep_Coral --run_description WISDM-Deep_Coral --da_method Deep_Coral --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-AdaMatch --run_description WISDM-AdaMatch --da_method AdaMatch --dataset WISDM
conda run -n cuda116 python -u main.py --experiment_description WISDM-DIRT --run_description WISDM-DIRT --da_method DIRT --dataset WISDM
