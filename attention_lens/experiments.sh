#!/bin/bash 

for model_name in /grand/SuperBERT/aswathy/models/models--meta-llama--Meta-Llama-3-8B-Instruct
do
    echo $model_name

    if [ "$model_name" == "gpt2" ]; then 
        num_layers=12
        job_name="gpt2_attnlen_"
    elif [ "$model_name" == "/grand/SuperBERT/aswathy/models/models--meta-llama--Meta-Llama-3-8B-Instruct" ]; then
        num_layers=32
        job_name="Llama3_attnlen_"
    else
        num_layers=36
        job_name="${model_name}_attnlen_"
    fi

    for (( layer=31; layer<$num_layers; layer++ ))
    do
        echo $layer
        ckpt_dir="/home/pettyjohnjn/AttentionLens/checkpoint3/${model_name}/ckpt_"
        qsub -v "ckpt=${job_name}${layer}, l_num=${layer}, model_name=$model_name" -N ${job_name}${layer} simple_submit.pbs 
    done

done