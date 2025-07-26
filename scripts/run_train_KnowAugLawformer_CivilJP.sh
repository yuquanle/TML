if true
# 修改：1）sh脚本参数；2）ini配置文件对应参数。
# sh ./train.sh
then
echo train model ...
# model_name: KnowAugLawformerCivilJP
# model_variants: LawformerCivilJP(input_max_length=1024)
# dataset_name: ['cpee']
dataset_name='cpee'
log_name='LawformerCivilJP_baseCLS_CivilJP_'${dataset_name}'_ep15_20220531.log'
echo log_name: ${log_name}
echo dataset_name: ${dataset_name}

# python -u ../trainerflow/KnowAugLawformerCivilJP_trainer.py  \
# --default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
# --dataset_name=${dataset_name} 

# nohup python -u ../trainerflow/KnowAugLawformerCivilJP_trainer.py \
# --default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
# --dataset_name=${dataset_name} > ../logs/${log_name} 2>& 1 &

# 分布式训练
# --nnode: 节点数目
# --nproc_per_node：每个节点进程数(GPU数目)
# --local_world_size：GPU数目
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 16681 ../trainerflow/KnowAugLawformerCivilJP_trainer.py \
--default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
--dataset_name=${dataset_name}

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 16689 ../trainerflow/KnowAugLawformerCivilJP_trainer.py \
# --default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
# --dataset_name=${dataset_name} > ../logs/${log_name} 2>& 1 &
fi


if false
# 修改：1）sh脚本参数；2）ini配置文件对应参数。
# sh ./train.sh
then
echo train model ...
# model_name: KnowAugLawformerCivilJP
# model_variants: LawformerCivilJP, KnowAugLawformerCivilJP_TD
# dataset_name: ['cpee']
dataset_name='cpee'
log_name='msBertMean_CivilJP_'${dataset_name}'_ep15_20230614.log'
echo log_name: ${log_name}
echo dataset_name: ${dataset_name}

# 分布式训练
# --nnode: 节点数目
# --nproc_per_node：每个节点进程数(GPU数目)
# --local_world_size：GPU数目
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 16675 ../trainerflow/KnowAugLawformerCivilJP_trainer.py \
# --default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
# --dataset_name=${dataset_name}

# CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 16688 ../trainerflow/KnowAugLawformerCivilJP_trainer.py \
# --default_config_path=../configs/msBert_CivilJPConfig.ini \
# --dataset_name=${dataset_name} > ../logs/${log_name} 2>& 1 &
fi