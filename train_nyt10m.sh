#!/bin/bash
export PYTHONUNBUFFERED=1
pretrain_path='bert-base-uncased'
train_file='benchmark/nyt10m/nyt10m_train.txt'
val_file='benchmark/nyt10m/nyt10m_val.txt'
test_file='benchmark/nyt10m/nyt10m_test.txt'
rel2id_file='benchmark/nyt10m/nyt10m_rel2id.json'
batch_size=48
max_length=128
max_epoch=3
seed=772
name=$pretrain_path'_'$max_length'_'$batch_size'_'$max_epoch'_'$seed'_nyt10m'
echo $name
python main.py --pretrain_path $pretrain_path --train_file $train_file --val_file $val_file --test_file $test_file --rel2id_file $rel2id_file --batch_size $batch_size --lr $lr --max_epoch $max_epoch --max_length $max_length --ckpt $name --devs 0