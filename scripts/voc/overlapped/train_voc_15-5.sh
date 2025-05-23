#!/bin/bash

GPU=5,6,7
BS=8
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for Adapter-M

NAME='Adapter'
OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json --port 22141 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json --port 22141 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}



