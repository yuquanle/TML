if false
# infer for Lawformer
# 修改：1）sh脚本参数；2）ini配置文件对应参数。
# sh ./infer.sh
then
echo infer model ...
# model_name: KnowAugLawformerCivilJP
# model_variants: LawformerCivilJP_baseCLS, LawformerCivilJP_baseMean, LawformerCivilJP, KnowAugLawformerCivilJP_TD
# dataset_name: ['cpee']
dataset_name='cpee'
echo dataset_name: ${dataset_name}

python -u ../inferflow/KnowAugLawformerCivilJP_infer.py  \
--default_config_path=../configs/KnowAugLawformer_CivilJPConfig.ini \
--dataset_name=${dataset_name} 

fi

if true
# infer for Bert-civil
# 修改：1）sh脚本参数；2）ini配置文件对应参数。
# sh ./infer.sh
then
echo infer model ...
# model_name: KnowAugLawformerCivilJP
# model_variants: msBertCLS, msBertMean
# dataset_name: ['cpee']
dataset_name='cpee'
echo dataset_name: ${dataset_name}

python -u ../inferflow/KnowAugLawformerCivilJP_infer.py  \
--default_config_path=../configs/msBert_CivilJPConfig.ini \
--dataset_name=${dataset_name} 

fi
