#!/bin/bash 

for model_name in gpt2
do
    echo $model_name

    job_name="${model_name}_attnlen_04_L"

    if [ $model_name == "gpt2" ]; then 
        declare -i num_layers=12
    else
        declare -i num_layers=36
    fi

    # Loop through all layers and skip layer 8
    for (( layer=0; layer<num_layers; layer++ ))
    do
        if [ $layer -eq 8 ]; then
            continue
        fi

        echo $layer
        ckpt_dir="/grand/SuperBERT/pettyjohnjn/AttnLens_GPT/checkpoint_rank04/${model_name}/ckpt_"
        qsub -v "ckpt=${ckpt_dir}${layer}, l_num=${layer}, model_name=$model_name" -N ${job_name}${layer} simple_submit.pbs 
    done

done