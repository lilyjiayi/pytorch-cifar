from http.client import NotConnected
import subprocess
from tkinter import NONE
import wandb


# excluded_machines = ','.join([f'jagupard{x}' for x in range(10, 26)])
# job_num = 452
# excluded_machines = "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20"
# add_on = ",jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27"
# excluded_machines = excluded_machines + add_on
# # launch_command = f"nlprun -g 1 -c 10 -x {excluded_machines} -n {job_num} -o ./out/{job_num}.out"
# launch_command = f"nlprun -g 1 -c 10 -q sphinx -n {job_num} -o ./out/{job_num}.out"
# id = None
# if id is None: id = wandb.util.generate_id()

# # imagenet
# # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 256 --dataset combined_imagenet --imagenet100 combined" \
# #                " --loader ffcv" \
# #                " --model resnet18" \
# #                " --drop_last --iweight_decay 1e-3 --weight_decay 1e-3" \
# #                " --inum_epoch 100 --num_epoch 100 --new_model --save_every 3" \
# #                " --ilr 0.05 --lr 0.05  --ischedule one_cycle  --schedule one_cycle" \
# #                " --seed_size 20000 --pool_size 100000 --query_size 10000 --num_queries 15" \
# #                " --group_div standard --exclude 0" \
# #                " --query_strategy random --score_fn MA --nocal" \
# #               f" --wandb_group combined_imagenet_al --wandb_name yfcc_seed_random --wandb_id {id} --save" \
# #                " --resume --checkpoint ./checkpoint/combined_imagenet_al-yfcc_seed_threshold-2pfw025f/0.pth" \

# # imagenet_10class
# # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset combined_imagenet --imagenet100 yfcc" \
# #                " --loader ffcv" \
# #                " --model resnet18 --pretrain" \
# #                " --iweight_decay 1e-4 --weight_decay 1e-4" \
# #                " --inum_epoch 100 --num_epoch 100 --new_model --save_every 10" \
# #                " --ilr 0.005 --lr 0.005  --ischedule cosine  --schedule cosine" \
# #                " --seed_size 200 --query_size 20 --num_queries 100" \
# #                " --group_div standard" \
# #                " --query_strategy random" \
# #               f" --wandb_group imagenet_10class_al_smallquery --wandb_name random --wandb_id {id} --save" \
# #             #    " --resume --checkpoint ./checkpoint/imagenet_10class_al-random_pretrain_lr5e-3-orp37wd6/0.pth" \

# # yfcc
# # base_command = "python ffcv_al.py --num_workers 8 --batch_size 128 --dataset geo_yfcc" \
# #                " --model resnet18" \
# #                " --frac 0.1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
# #                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
# #                " --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine" \
# #                " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 30" \
# #                " --group_div standard" \
# #                " --query_strategy random" \
# #               f" --wandb_group geo_yfcc_10k --wandb_name random_nopre --wandb_id {id} --save" 


# # domainnet four non-pretrained
# # base_command = "python al.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry" \
# #                " --model resnet18" \
# #                " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
# #                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 20" \
# #                " --ilr 5e-2 --lr 5e-2  --ischedule cosine  --schedule cosine" \
# #                " --seed_size 1000 --pool_size 40000 --query_size 50 --num_queries 150" \
# #                " --group_div standard" \
# #                " --query_strategy threshold --score_fn NE --nocal" \
# #               f" --wandb_group domainnet_1k_50_nopre --wandb_name threshold_entropy_nocal --wandb_id {id} --save" \
# #                " --resume --checkpoint ./checkpoint/domainnet_1k_50_nopre-lr_5e-2-nx6zufox/0.pth"

# # domainnet four pretrained
# # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry" \
# #                " --model resnet18 --pretrain" \
# #                " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
# #                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
# #                " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
# #                " --seed_size 1000 --pool_size 40000 --query_size 50 --num_queries 100" \
# #                " --group_div standard" \
# #                " --query_strategy threshold" \
# #               f" --wandb_group domainnet_1k_50_rerun --wandb_name threshold_pool --wandb_id {id} --save" \
# #                " --resume --checkpoint ./checkpoint/domainnet_1k_50_rerun-least_confidence-1v1r98db/0.pth"

# # domainnet six
# base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet" \
#                " --model resnet18 --pretrain" \
#                " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
#                " --inum_epoch 50 --num_epoch 50 --new_model --save_every 5" \
#                " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
#                " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 50" \
#                " --group_div standard" \
#                " --group_strategy loss_exp --query_strategy random" \
#               f" --wandb_group domainnet_10k_500_rerun --wandb_name loss_exp_random --wandb_id {id} --save" \
#                " --resume --checkpoint ./checkpoint/domainnet_10k_500_rerun-random-1lnj092i/0.pth"

# command = f"{launch_command} '{base_command}'"
# print(f"Launching command: {command}")
# subprocess.run(command, shell=True)


def run_expt(group, query, job_num, id = None, name = None, distribution = None, counts = None, checkpoint = None, score_ma = False, p_high = False):
  excluded_machines = ','.join([f'jagupard{x}' for x in range(10, 26)])
  job_num = job_num
  excluded_machines = "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20"
  add_on = ",jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27"
  excluded_machines = excluded_machines + add_on
  priority = " -p high" if p_high else ""
  launch_command = f"nlprun -r 50g -g 1 -c 10 -d a6000 -n {job_num} -o ./out/{job_num}.out" + priority
  # launch_command = f"nlprun -r 50g -g 1 -c 10 -x {excluded_machines} -n {job_num} -o ./out/{job_num}.out" + priority
  # launch_command = f"nlprun -g 1 -c 10 -n {job_num} -o ./out/{job_num}.out" + priority
  # launch_command = f"nlprun -r 50g -g 1 -c 10 -q sphinx -x sphinx3 -n {job_num} -o ./out/{job_num}.out" + priority
  if id is None: id = wandb.util.generate_id()
  # if name is None: name = f"{group}_{query}"

  if group is None:
    # query_command = f" --query_strategy {query} --score_fn MA"
    query_command = f" --query_strategy {query}"
    if score_ma: query_command += ' --score_fn MA'
    if name is None: name = f"{query}"
  else:
    # query_command = f" --group_strategy {group} --query_strategy {query} --score_fn MA"
    query_command = f" --group_strategy {group} --query_strategy {query}"
    if score_ma: query_command += ' --score_fn MA'
    if name is None: name = f"{group}_{query}"

  if checkpoint is None: 
    checkpoint_command = ""
  else:
    checkpoint_command = f" --resume --checkpoint {checkpoint}"

  # if query == 'random':
  #   num_query = 150
  # else:
  #   num_query = 50
    
  num_query = 50

  if distribution is not None: 
    sample_command = f'--include {distribution}'
  elif counts is not None:
    sample_command = f'--include_counts {counts}'
  else:
    sample_command = ''

  # domainnet four
  # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_four" \
  #              " --model resnet50 --pretrain" \
  #              " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #              " --inum_epoch 50 --num_epoch 50 --new_model --save_every 5" \
  #              " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
  #              " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 50" \
  #              " --group_div standard" \
  #             f"{query_command}" \
  #             f" --wandb_group domainnet_four_10k_500 --wandb_name {name} --wandb_id {id} --save" \
  #             " --resume --checkpoint ./checkpoint/domainnet_four_10k_500-random_resnet50-1hgr458q/0.pth"
  #             #  " --resume --checkpoint ./checkpoint/domainnet_four_10k_500-random-msrjml9f/0.pth"
  #             #  " --resume --checkpoint ./checkpoint/domainnet_four_10k_500-random-2edl0su1/0.pth" # old transform

  # geoyfcc superclass 5
  # base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset geo_yfcc --geoyfcc_num_class 45 --superclass 5 {sample_command}" \
  #               " --model resnet18 --pretrain" \
  #               " --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #               " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #               " --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine" \
  #               f" --seed_size 1000  --pool_size 40000 --query_size 50 --num_queries {num_query}" \
  #               f"{query_command}" \
  #               f" --wandb_group geoyfcc_superclass_5 --wandb_name {name} --wandb_id {id} --save" \
  #               f"{checkpoint_command}"

  # geoyfcc 45
  # base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset geo_yfcc --geoyfcc_num_class 45 {sample_command}" \
  #               " --model resnet18 --pretrain" \
  #               " --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #               " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #               " --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine" \
  #               f" --seed_size 1000  --pool_size 40000 --query_size 50 --num_queries {num_query}" \
  #               f"{query_command}" \
  #               f" --wandb_group geoyfcc_45_mixture --wandb_name {name} --wandb_id {id} --save" \
  #               f"{checkpoint_command}"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_noafrica_45-random-21729wwy/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_1k_50-random-soqk5ica/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_10k_1k-random_350-110wbrdz/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc-random_350-35r4pg70/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_10k_500-random_500-2wqru27r/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_10k_500-random_350-32wsthaw/0.pth"
  #               # " --resume --checkpoint ./checkpoint/geoyfcc_10k_1k-random_500-2qn881a2/0.pth"
  
  # geoyfcc 350
  # base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset geo_yfcc --geoyfcc_num_class 350 {sample_command}" \
  #               " --model resnet18 --pretrain" \
  #               " --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #               " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #               " --ilr 2e-2 --lr 2e-2  --ischedule cosine  --schedule cosine" \
  #               f" --seed_size 10000  --pool_size 40000 --query_size 1000 --num_queries 50" \
  #               f"{query_command}" \
  #               f" --wandb_group geoyfcc_10k_1k --wandb_name {name} --wandb_id {id} --save" \
  #               f"{checkpoint_command}"

  # # domainnet six with 40 classes
  # base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_40_class {sample_command}" \
  #              " --model resnet18 --pretrain" \
  #              " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #              " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #              " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
  #              f" --seed_size 1000 --pool_size 0 --query_size 50 --num_queries {num_query}" \
  #              " --group_div standard" \
  #              f"{query_command}" \
  #              f" --wandb_group domainnet_40_mixture --wandb_name {name} --wandb_id {id} --save" \
  #              f"{checkpoint_command}"    
  
  # domainnet six with 40 classes with randomization
  base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_40_class {sample_command}" \
               " --model resnet18 --pretrain" \
               " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
               " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
               " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
               f" --seed_size 1000 --pool_size 1000 --query_size 50 --num_queries {num_query}" \
               " --group_div standard" \
               f"{query_command}" \
               f" --wandb_group domainnet_40_mixture_randomized --wandb_name {name} --wandb_id {id} --save" \
               f"{checkpoint_command}"   
  #  f" --wandb_group domainnet_40_mixture --wandb_name {name} --wandb_id {id} --save" \


  # unbalanced domainnet sentry
  # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry --subsampled" \
  #              " --model resnet18 --pretrain" \
  #              " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #              " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #              " --ilr 1e-2 --lr 1e-2  --ischedule cosine  --schedule cosine" \
  #              " --seed_size 1000 --pool_size 40000 --query_size 50 --num_queries 100" \
  #              " --group_div standard" \
  #             f"{query_command}" \
  #             f" --wandb_group unbalanced_domainnet_1k_50 --wandb_name {name} --wandb_id {id} --save" \
  #             " --resume --checkpoint ./checkpoint/unbalanced_domainnet_1k_50-random-2usmsmuo/0.pth"

  # domainnet six
  # base_command = f"python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet {sample_command}" \
  #              " --model resnet18 --pretrain" \
  #              " --frac 0.05 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #              " --inum_epoch 50 --num_epoch 50 --new_model --save_every 5" \
  #              " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
  #              " --seed_size 10000 --pool_size 40000 --query_size 500 --num_queries 50" \
  #              " --group_div standard" \
  #             f"{query_command}" \
  #             f" --wandb_group domainnet_10k_500_rerun --wandb_name {name} --wandb_id {id} --save" \
  #             f"{checkpoint_command}"
  #             #  " --resume --checkpoint ./checkpoint/domainnet_10k_500_rerun-random-1lnj092i/0.pth"
  
  # domainnet sentry
  # base_command = "python ffcv_al_script.py --num_workers 8 --batch_size 128 --dataset domainnet --use_sentry" \
  #              " --model resnet18 --pretrain" \
  #              " --frac 1 --drop_last --iweight_decay 1e-4 --weight_decay 1e-4" \
  #              " --inum_epoch 50 --num_epoch 50 --new_model --save_every 10" \
  #              " --ilr 5e-3 --lr 5e-3  --ischedule cosine  --schedule cosine" \
  #              " --seed_size 1000 --pool_size 10000 --query_size 50 --num_queries 100" \
  #              " --group_div standard" \
  #             f"{query_command}" \
  #             f" --wandb_group domainnet_1k_50_rerun --wandb_name {name} --wandb_id {id} --save" \
  #             f"{checkpoint_command}"
              # " --resume --checkpoint ./checkpoint/domainnet_1k_50_rerun-random_new_transform-2x0p4tpc/0.pth" 
              # " --resume --checkpoint ./checkpoint/domainnet_1k_50_rerun-least_confidence-1v1r98db/0.pth"
  
  command = f"{launch_command} '{base_command}'"
  print(f"Launching command: {command}")
  subprocess.run(command, shell=True)


# job_num = 600
# # checkpoint = './checkpoint/domainnet_40_mixture-random_clip_quick_real-2denf8ag/0.pth'
# checkpoint = None
# run_expt(None, 'random', job_num, id = None, name = 'random_1356', distribution = '1,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 601
# checkpoint = './checkpoint/domainnet_10k_500_rerun-random_1356-1sqmmufw/0.pth'
# run_expt(None, 'margin', job_num, id = None, name = 'margin_1356', distribution = '1,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)


# 12345
# 356
# 145
# 234
# 345
# 2456
# 1245
# 1236
# 1346
# 12346
# 13456
# 12356

# job_num = 643
# checkpoint = './checkpoint/domainnet_40_mixture-random_noquick_sketch-ueofyv2v/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1235', distribution = '1,1,1,0,1,0', score_ma = True, checkpoint=checkpoint)
  
# job_num = 645
# checkpoint = './checkpoint/domainnet_40_mixture-random_nopaint_real-2xxweimk/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1246', distribution = '1,1,0,1,0,1', score_ma = True, checkpoint=checkpoint)



# domainnet 40 with randomization
# checkpoint = None
# # job_num = 639
# # # checkpoint = './checkpoint/domainnet_40_mixture-random_clip_quick_real-2denf8ag/0.pth'
# # run_expt(None, 'random', job_num, id = None, name = 'random_145', distribution = '1,0,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# checkpoint = None
# job_num = 640
# # checkpoint = './checkpoint/domainnet_40_mixture-random_paint_real_sketch-2ajv3odb/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_356', distribution = '0,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 641
# # checkpoint = './checkpoint/domainnet_40_mixture-random_paint_quick_real-2agtvwte/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_345', distribution = '0,0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 642
# # checkpoint = './checkpoint/domainnet_40_mixture-random_info_paint_quick-24cb7b9o/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_234', distribution = '0,1,1,1,0,0', score_ma = True, checkpoint=checkpoint)

# job_num = 643
# # checkpoint = './checkpoint/domainnet_40_mixture-random_nosketch-2z6pbn2k/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_12345', distribution = '1,1,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 644
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noclip_paint-22uay2y6/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_2456', distribution = '0,1,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 645
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noquickdraw_real-3tl5gufy/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_1236', distribution = '1,1,1,0,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 646
# # checkpoint = './checkpoint/domainnet_40_mixture-random_nopainting_sketch-37uhigat/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_1245', distribution = '1,1,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 647
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noinfograph_real-wqlybgna/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_1346', distribution = '1,0,1,1,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 648
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noreal-w1qyse1t/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_12346', distribution = '1,1,1,1,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 649
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noinfograph-3dobrdvk/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_13456', distribution = '1,0,1,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 650
# # checkpoint = './checkpoint/domainnet_40_mixture-random_noquickdraw-18bvjd9e/0.pth'
# run_expt(None, 'random', job_num, id = None, name = 'random_12356', distribution = '1,1,1,0,1,1', score_ma = True, checkpoint=checkpoint)


############## domainnet_40_randomized margin
job_num = 671
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_145-2hoihxgg/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_145', distribution = '1,0,0,1,1,0', score_ma = True, checkpoint=checkpoint)


job_num = 672
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_356-17m090dk/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_356', distribution = '0,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)

job_num = 673
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_345-2ilvpeym/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_345', distribution = '0,0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

job_num = 674
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_234-2chkr3na/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_234', distribution = '0,1,1,1,0,0', score_ma = True, checkpoint=checkpoint)

job_num = 675
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_12345-y5fpo7tt/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_12345', distribution = '1,1,1,1,1,0', score_ma = True, checkpoint=checkpoint)

job_num = 676
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_2456-zwugc1og/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_2456', distribution = '0,1,0,1,1,1', score_ma = True, checkpoint=checkpoint)

job_num = 677
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_1236-1jtbumhc/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_1236', distribution = '1,1,1,0,0,1', score_ma = True, checkpoint=checkpoint)

job_num = 678
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_1245-3vj8h9no/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_1245', distribution = '1,1,0,1,1,0', score_ma = True, checkpoint=checkpoint)

job_num = 679
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_1346-2gei0835/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_1346', distribution = '1,0,1,1,0,1', score_ma = True, checkpoint=checkpoint)

job_num = 680
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_12346-1kk2kjft/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_12346', distribution = '1,1,1,1,0,1', score_ma = True, checkpoint=checkpoint)

job_num = 681
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_13456-2d9g7ekv/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_13456', distribution = '1,0,1,1,1,1', score_ma = True, checkpoint=checkpoint)

job_num = 682
checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_12356-2v12v68r/0.pth'
run_expt(None, 'margin', job_num, id = None, name = 'margin2_12356', distribution = '1,1,1,0,1,1', score_ma = True, checkpoint=checkpoint)

###### second run
# checkpoint = None
# job_num = 663
# checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_356-17m090dk/0.pth'
# run_expt(None, 'margin', job_num, id = None, name = 'margin_356', distribution = '0,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)
  
# job_num = 664
# checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_345-2ilvpeym/0.pth'
# run_expt(None, 'margin', job_num, id = None, name = 'margin_345', distribution = '0,0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 665
# checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_234-2chkr3na/0.pth'
# run_expt(None, 'margin', job_num, id = None, name = 'margin_234', distribution = '0,1,1,1,0,0', score_ma = True, checkpoint=checkpoint)

# job_num = 666
# checkpoint = './checkpoint/domainnet_40_mixture_randomized-random_12345-y5fpo7tt/0.pth'
# run_expt(None, 'margin', job_num, id = None, name = 'margin_12345', distribution = '1,1,1,1,1,0', score_ma = True, checkpoint=checkpoint)


# domainnet 40 threshold group
# job_num = 639
## checkpoint = './checkpoint/domainnet_40_mixture-random_clip_quick_real-2denf8ag/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_145', distribution = '1,0,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 640
## checkpoint = './checkpoint/domainnet_40_mixture-random_paint_real_sketch-2ajv3odb/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_356', distribution = '0,0,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 641
## checkpoint = './checkpoint/domainnet_40_mixture-random_paint_quick_real-2agtvwte/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_345', distribution = '0,0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 642
## checkpoint = './checkpoint/domainnet_40_mixture-random_info_paint_quick-24cb7b9o/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_234', distribution = '0,1,1,1,0,0', score_ma = True, checkpoint=checkpoint)

# job_num = 643
# checkpoint = './checkpoint/domainnet_40_mixture-random_noquick_sketch-ueofyv2v/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1235', distribution = '1,1,1,0,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 644
# checkpoint = './checkpoint/domainnet_40_mixture-random_nosketch-2z6pbn2k/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_12345', distribution = '1,1,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 645
# checkpoint = './checkpoint/domainnet_40_mixture-random_nopaint_real-2xxweimk/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1246', distribution = '1,1,0,1,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 646
# checkpoint = './checkpoint/domainnet_40_mixture-random_noclip_paint-22uay2y6/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_2456', distribution = '0,1,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 647
# checkpoint = './checkpoint/domainnet_40_mixture-random_noquickdraw_real-3tl5gufy/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1236', distribution = '1,1,1,0,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 648
# checkpoint = './checkpoint/domainnet_40_mixture-random_nopainting_sketch-37uhigat/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1245', distribution = '1,1,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 649
# checkpoint = './checkpoint/domainnet_40_mixture-random_noinfograph_real-wqlybgna/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_1346', distribution = '1,0,1,1,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 650
# checkpoint = './checkpoint/domainnet_40_mixture-random_noreal-w1qyse1t/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_12346', distribution = '1,1,1,1,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 651
# checkpoint = './checkpoint/domainnet_40_mixture-random_noinfograph-3dobrdvk/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_13456', distribution = '1,0,1,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 652
# checkpoint = './checkpoint/domainnet_40_mixture-random_noquickdraw-18bvjd9e/0.pth'
# run_expt('error_prop', 'margin', job_num, id = None, name = 'error_prop_margin_12356', distribution = '1,1,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# geoyfcc 45 threshold group
# job_num = 603 
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_1234-25wi0fbj/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1234', distribution = '1,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 604
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_234-2qlonwfv/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_234', distribution = '0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 605
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_345-1zvgud9j/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_345', distribution = '0,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 606
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_125-1gz4kl5c/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_125', distribution = '1,1,0,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 607
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_134-1d9tdkng/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_134', distribution = '1,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 608
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_145-196cuhih/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_145', distribution = '1,0,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 609
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_1245-2oirrgnv/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1245', distribution = '1,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 610
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_1345-1wgp6dli/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1345', distribution = '1,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 611
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_2345-2sr7co3j/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_2345', distribution = '0,1,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 612
# checkpoint = './checkpoint/geoyfcc_45_mixture-random_123-66ixq61i/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_123', distribution = '1,1,1,0,0', score_ma = True, checkpoint=checkpoint)

# geoyfcc 5 threshold group
# job_num = 654
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_234-3n2hnhrm/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_234', distribution = '0,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 655
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_345-2ne6io07/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_345', distribution = '0,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 656
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_125-38eslkqh/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_125', distribution = '1,1,0,0,1', score_ma = True, checkpoint=checkpoint)

# job_num = 657
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_134-26wa4n7o/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_134', distribution = '1,0,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 658
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_145-13wok4vt/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_145', distribution = '1,0,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 659
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_1245-3lymhv1z/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1245', distribution = '1,1,0,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 660
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_1345-2amzc93g/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1345', distribution = '1,0,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 661
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_2345-1ulshh0s/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_2345', distribution = '0,1,1,1,1', score_ma = True, checkpoint=checkpoint)

# job_num = 662
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_1234-17voapzk/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_1234', distribution = '1,1,1,1,0', score_ma = True, checkpoint=checkpoint)

# job_num = 663
# checkpoint = './checkpoint/geoyfcc_superclass_5-random_123-n2dyphc7/0.pth'
# run_expt('uniform', 'margin', job_num, id = None, name = 'uniform_margin_123', distribution = '1,1,1,0,0', score_ma = True, checkpoint=checkpoint)




# job_num = 540
# for i in range(5):
#   distribution = [0] * 5
#   distribution[i] = 1
#   distribution = [str(n) for n in distribution]
#   distribution = ','.join(distribution)
#   run_expt(None, 'random', job_num, id = None, name = f'350_{i}', distribution = distribution)
#   job_num += 1


# group_strategy = ['loss_exp']
# query_strategy = ['margin', 'random']

# job_num = 630
# i = 0
# group_strategy = ['uniform', 'loss_prop', 'error_prop', 'loss_exp', 'oracle']
# query_strategy = ['least_confidence', 'entropy']
# # checkpoint = './checkpoint/domainnet_1k_50_rerun-least_confidence-1v1r98db/0.pth'
# checkpoint = './checkpoint/geoyfcc_1k_50-random-soqk5ica/0.pth'
# # checkpoint = './checkpoint/geoyfcc_10k_1k-random_350-110wbrdz/0.pth'
# # checkpoint = None

# for group in group_strategy:
#   for query in query_strategy:
#     # if query == 'least_confidence': 
#     #   p_high = True
#     # else:
#     #   p_high = False
#     p_high = False
#     run_expt(group, query, job_num, id = None, checkpoint=checkpoint, p_high=p_high)
#     job_num += 1
#     # i += 1

# job_num = 641
# run_expt(None, 'least_confidence', job_num, id = None, checkpoint=checkpoint, p_high=True)

# job_num = 642
# run_expt(None, 'entropy', job_num, id = None, checkpoint=checkpoint, p_high=True)




