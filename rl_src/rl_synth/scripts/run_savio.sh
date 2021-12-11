launch_mbrl_train_job(){
    BATCH_INITIAL=$1
    AGENT_TRAIN_STEPS=$2
    TRAINING_BATCH=$3
    NN_SIZE=$4
    NN_LAYERS=$5
    slurm_script_name="mbrl_train_batchinit$1_natspi$2_tb$3_model$4x$5"
    echo $slurm_script_name
    cat > "${slurm_script_name}.sh" <<EOT
#!/bin/bash
# generated at $(date)
# Job name:
#SBATCH --job-name=${slurm_script_name}
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio
#
# Quality of Service:
#SBATCH --qos=savio_normal
# Num Cores per Task
#SBATCH --cpus-per-task=4
#
# Wall clock limit:
#SBATCH --time=05:00:00
#
## Command(s) to run:
source /global/home/users/${USER}/.bashrc
source activate rl
python ../rl_synth/scripts/run.py --exp_name  ${slurm_script_name} --env_name synthesis-v0 \
        --add_sl_noise --n_iter 1 \
        --batch_size_initial ${1} --num_agent_train_steps_per_iter ${2} --train_batch_size ${3}\
        --n_layers ${NN_LAYERS} --size ${NN_SIZE} --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'    
EOT
    sbatch "${slurm_script_name}.sh"
}
if ! [ -d "runs" ]; then
  mkdir -p "runs"
fi
pushd runs

declare -a batchinit_array=("500" "1000" )
declare -a natspi_array=("5" "50" )
declare -a tb_array=("5" "50" )
declare -a nn_size_array=("8" "16" "128" )
declare -a nn_layer_array=("1" "2" )

for batchinit in "${batchinit_array[@]}"
do 
    for natspi in "${natspi_array[@]}"
    do 
        for tb in "${tb_array[@]}" 
        do 
            for nnsize in "${nn_size_array[@]}"
            do 
                for nnlayer in "${nn_layer_array[@]}"
                do
                    launch_mbrl_train_job $batchinit $natspi $tb $nnsize $nnlayer           
                done
            done
        done
    done
done
