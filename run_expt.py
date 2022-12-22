import subprocess
import wandb

excluded_machines = ','.join([f'jagupard{x}' for x in range(10, 26)])
job_num = 380
excluded_machines = "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20"
add_on = ",jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27"
excluded_machines = excluded_machines + add_on
# launch_command = f"nlprun -g 1 -c 10 -x {excluded_machines} -n {job_num} -o ./out/{job_num}.out"
launch_command = f"nlprun -g 1 -c 12 -m sphinx2 -r 50g -n {job_num} -o ./out/{job_num}.out"
id = 'czuq3ett'
if id is None: id = wandb.util.generate_id()

# imagenet
base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset imagenet --log_op no_log" \
               " --loader ffcv" \
               " --model resnet18" \
               " --frac 1 --drop_last --iweight_decay 5e-5 --weight_decay 5e-5" \
               " --inum_epoch 50 --num_epoch 100 --new_model --save_every 5" \
               " --ilr 0.2 --lr 0.2  --ischedule one_cycle  --schedule one_cycle" \
               " --seed_size 100000 --pool_size 100000 --query_size 10000 --num_queries 40" \
               " --group_div standard" \
               " --query_strategy threshold --score_fn MC --nocal" \
               " --resume --checkpoint ./checkpoint/imagenet_10k-random_lr0.2_ep50-2svlnbhb/0.pth" \
              f" --wandb_group imagenet_10k --wandb_name threshold_lc_one_cycle --wandb_id {id} --save" \

# yfcc
# base_command = "python ffcv_al.py --num_workers 8 --batch_size 128 --dataset geo_yfcc" \
#                " --model resnet18" \
#                " --frac 0.1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
#                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
#                " --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine" \
#                " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 30" \
#                " --group_div standard" \
#                " --query_strategy random" \
#               f" --wandb_group geo_yfcc_10k --wandb_name random_nopre --wandb_id {id} --save" 


# domainnet four non-pretrained
# base_command = "python al.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry" \
#                " --model resnet18" \
#                " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
#                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 20" \
#                " --ilr 5e-2 --lr 5e-2  --ischedule cosine  --schedule cosine" \
#                " --seed_size 1000 --pool_size 40000 --query_size 50 --num_queries 150" \
#                " --group_div standard" \
#                " --query_strategy threshold --score_fn NE --nocal" \
#               f" --wandb_group domainnet_1k_50_nopre --wandb_name threshold_entropy_nocal --wandb_id {id} --save" \
#                " --resume --checkpoint ./checkpoint/domainnet_1k_50_nopre-lr_5e-2-nx6zufox/0.pth"

# domainnet four pretrained
# base_command = "python al.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry" \
#                " --model resnet18 --pretrain" \
#                " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
#                " --inum_epoch 100 --num_epoch 50 --new_model --save_every 20" \
#                " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
#                " --seed_size 1000 --pool_size 40000 --query_size 50 --num_queries 120" \
#                " --group_div standard" \
#                " --group_strategy uniform --query_strategy least_confidence --noise 2" \
#               f" --wandb_group domainnet_1k_50 --wandb_name uniform_uncertainty_n2 --wandb_id {id} --save" \
#                " --resume --checkpoint ./checkpoint/domainnet_1k_50-oracle_seed-pf4v5qan/0.pth"

# domainnet six
# base_command = "python al.py --num_workers 8 --batch_size 128 --dataset domainnet" \
#                " --model resnet18 --pretrain" \
#                " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
#                " --inum_epoch 100 --num_epoch 50 --new_model --save_every 5" \
#                " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
#                " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 40" \
#                " --group_div standard" \
#                " --group_strategy uniform --query_strategy least_confidence --noise 0.5" \
#               f" --wandb_group domainnet_10k_500_new --wandb_name uniform_uncertainty_n0.5 --wandb_id {id} --save" \
#                " --resume --checkpoint ./checkpoint/domainnet_10k_500_new-random-281wtpzw/0.pth"

command = f"{launch_command} '{base_command}'"
print(f"Launching command: {command}")
subprocess.run(command, shell=True)


