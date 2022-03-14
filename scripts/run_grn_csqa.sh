#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 48:00:00
# dev acc: 0.8010

N=$GPU_DEVICE_ORDINAL


dataset="obqa"
kb="all"
min_path_length=1
max_path_length=5
max_num_paths=10
max_node=400
model='roberta-large'
dt=`date '+%Y%m%d_%H%M%S'`
k=1
seed=0
num_relations=42
ent_emb="tzw"


if [[ ${dataset} = "csqa" ]]
then
    ih=1
else
    ih=0
fi


CUDA_VISIBLE_DEVICES=$N python3 -u grn.py -k ${k} \
        --unfreeze_epoch 3 \
        --format fairseq \
        --fix_trans \
        --ent_emb ${ent_emb} \
        -ih ${ih} \
        -enc ${model} \
        -ds ${dataset} \
        -mbs 8 \
        -sl 80 \
        -me 10 \
        --seed ${seed} \
        --kb ${kb} \
        --min_path_length ${min_path_length} \
        --max_path_length ${max_path_length} \
        --max_num_paths ${max_num_paths} \
        --max_node_num ${max_node} \
        --ent_emb_paths "./data/${kb}/tzw.ent.npy" \
        --train_statements "./data/${dataset}_${kb}/statement/train.statement.jsonl" \
        --dev_statements "./data/${dataset}_${kb}/statement/dev.statement.jsonl" \
        --test_statements "./data/${dataset}_${kb}/statement/test.statement.jsonl" \
        --inhouse_train_qids "./data/${dataset}_${kb}/inhouse_split_qids.txt" \
        --num_relation ${num_relations} \
        > train_${kb}_${dataset}_max_nodes_${max_node}_emb_${ent_emb}_enc-${model}_k_${k}_seed_${seed}_${dt}.log.txt
        #> train_${kb}_${dataset}_min_length_${min_path_length}_max_length_${max_path_length}_max_paths_${max_num_paths}_emb_${ent_emb}_enc-${model}_k_${k}_seed_${seed}_${dt}.log.txt
