# group AL

Active Learning methods on celebA, domainnet, and geo-yfcc

## For combined_imagenet FFCV training 
```
conda activate jl_ffcv

# Train on 100-class yfcc_imagenet (100k), no active learning 
python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet --imagenet100 yfcc \
--loader ffcv --model resnet18 --frac 0.01 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5 \
--inum_epoch 100 --num_epoch 100 --ilr 0.5 --lr 0.5  --ischedule one_cycle --schedule one_cycle \
--new_model --save_every 2 --seed_size 100000 --pool_size 100000 --query_size 10000 \
--num_queries 0 --group_div standard --query_strategy random

# Train on 100-class imagenet (100k), no active learning 
python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet --imagenet100 imagenet \ 
--loader ffcv --model resnet18 --frac 0.01 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5 \
--inum_epoch 100 --num_epoch 100 --ilr 0.5 --lr 0.5  --ischedule one_cycle --schedule one_cycle \
--new_model --save_every 2 --seed_size 100000 --pool_size 100000 --query_size 10000 \ 
--num_queries 0 --group_div standard --query_strategy random

# Train on 100-class combined_imagenet (200k), no active learning 
python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet --imagenet100 combined \
--loader ffcv --model resnet18 --frac 0.01 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5 \
--inum_epoch 100 --num_epoch 100 --ilr 0.5 --lr 0.5  --ischedule one_cycle --schedule one_cycle \
--new_model --save_every 2 --seed_size 200000 --pool_size 100000 --query_size 10000 \
--num_queries 0 --group_div standard --query_strategy random

# We can also train active learning loop on 100-class data, just need to make sure there are enough samples
python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet --imagenet100 combined \
--loader ffcv --model resnet18 --frac 0.01 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5 \
--inum_epoch 100 --num_epoch 100 --ilr 0.5 --lr 0.5  --ischedule one_cycle --schedule one_cycle \
--new_model --save_every 2 --seed_size 25000 --pool_size 50000 --query_size 10000 \
--num_queries 20 --group_div standard --query_strategy random

# Train on original combined_imagnet, active learning loop
python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet \
--loader ffcv --model resnet18 --frac 0.01 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5 \
--inum_epoch 100 --num_epoch 100 --ilr 0.5 --lr 0.5  --ischedule one_cycle --schedule one_cycle \
--new_model --save_every 2 --seed_size 100000 --pool_size 100000 --query_size 10000 \
--num_queries 20 --group_div standard --query_strategy random

```



## For FFCV compatible training
```
conda activate jl_ffcv

python ffcv_al.py --dataset geo_yfcc --frac 0.1 \
 --model resnet18 --pretrain \
 --loader ffcv --num_workers 8 --batch_size 128 --drop_last \
 --iweight_decay 1e-4 --weight_decay 1e-4 \
 --inum_epoch 50 --num_epoch 50 --new_model \
 --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine \
 --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 30 \
 --group_div standard --query_strategy random \
```

## Prerequisites
#conda env created in the public conda directory

conda activate jl_ffcv

## Training
```
# normal 
python al.py --dataset celebA --model resnet18 --drop_last --inum_epoch 100 --num_epoch 10 --batch_size 128 --ilr 1e-4 --lr 1e-5 --ischedule cosine  --schedule cosine --num_workers 8 --query_strategy least_confidence --seed_size 10000 --num_queries 300 --query_size 500 --wandb_group celebA_10k_500_iep100_normal --save

# random
python al.py --dataset celebA --model resnet18 --drop_last --inum_epoch 100 --num_epoch 10 --batch_size 128 --ilr 1e-4 --lr 1e-5 --ischedule cosine  --schedule cosine --num_workers 8 --query_strategy random --seed_size 10000 --num_queries 300 --query_size 500 --wandb_group celebA_10k_500_iep100_random --save

# oracle 
python al.py --dataset celebA --model resnet18 --drop_last --inum_epoch 100 --num_epoch 10 --batch_size 128 --ilr 1e-4 --lr 1e-5 --ischedule cosine  --schedule cosine --num_workers 8 --group_strategy oracle --query_strategy random --seed_size 10000 --num_queries 300 --query_size 500 --wandb_group celebA_10k_500_iep100_oracle --save

# avg_c  
python al.py --dataset celebA --model resnet18 --drop_last --inum_epoch 100 --num_epoch 10 --batch_size 128 --ilr 1e-4 --lr 1e-5 --ischedule cosine  --schedule cosine --num_workers 8 --group_strategy avg_c_val --query_strategy random --seed_size 10000 --num_queries 300 --query_size 500 --wandb_group celebA_10k_500_iep100_avgc --save

# min
python al.py --dataset celebA --model resnet18 --drop_last --inum_epoch 100 --num_epoch 10 --batch_size 128 --ilr 1e-4 --lr 1e-5 --ischedule cosine  --schedule cosine --num_workers 8 --group_strategy min --query_strategy random --seed_size 10000 --num_queries 300 --query_size 500 --wandb_group celebA_10k_500_iep100_min --save
```





