#!/bin/bash 

model_name="gpt2"
lora_rank="12"
echo "$model_name"

job_name="${model_name}_attnlen_R${lora_rank}_all"

ckpt_dir="/grand/SuperBERT/pettyjohnjn/LoRA_Lens/checkpoint/full/R${lora_rank}/${model_name}/ckpt_"

qsub -v "ckpt=${ckpt_dir},model_name=${model_name},lora_rank=${lora_rank}" -N "${job_name}" simple_submit.pbs