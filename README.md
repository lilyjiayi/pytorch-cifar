# group AL

Different Active Learning methods on celebA and waterbirds

## Prerequisites
#conda env created in the public conda directory

conda activate jiayili_al

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



