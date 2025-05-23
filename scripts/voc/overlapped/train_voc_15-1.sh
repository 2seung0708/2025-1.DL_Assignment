#!/bin/bash

GPU=0,1,2
BS=8
SAVEDIR='saved_voc'
MASTER_PORT=12345 ## added for multi-GPU training

TASKSETTING='overlap'
TASKNAME='15-1'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for Adapter-M

NAME='Adapter'
OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

OMP_NUM_THREADS=8 python train_voc.py -c configs/config_voc.json  --port 12133 \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}


