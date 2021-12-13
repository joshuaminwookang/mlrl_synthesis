launch_mbrl_train_job(){
    BATCH_INITIAL=$1
    AGENT_TRAIN_STEPS=$2
    TRAINING_BATCH=$3
    NN_SIZE=$4
    NN_LAYERS=$5
    LEARNING_RATE=$6
    slurm_script_name="mbrl_train_batchinit$1_natspi$2_tb$3_model$4x$5_lr$6"
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
#SBATCH --partition=savio2
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
        --batch_size_initial ${BATCH_INITIAL} --num_agent_train_steps_per_iter ${AGENT_TRAIN_STEPS}\
        --train_batch_size ${TRAINING_BATCH} --n_iter 1\
        --n_layers ${NN_LAYERS} --size ${NN_SIZE} --scalar_log_freq 1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'  --learning_rate ${LEARNING_RATE} 
EOT
    sbatch "${slurm_script_name}.sh"
}

launch_mbrl_run_job(){
    BATCH_INITIAL=$1
    AGENT_TRAIN_STEPS=$2
    TRAINING_BATCH=$3
    NN_SIZE=$4
    NN_LAYERS=$5
    LEARNING_RATE=$6
    slurm_script_name="mbrl_train_batchinit$1_natspi$2_tb$3_model$4x$5_lr$6"
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
#SBATCH --partition=savio2
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
        --batch_size_initial ${BATCH_INITIAL} --num_agent_train_steps_per_iter ${AGENT_TRAIN_STEPS}\
        --train_batch_size ${TRAINING_BATCH} --n_iter 1\
        --n_layers ${NN_LAYERS} --size ${NN_SIZE} --scalar_log_freq 1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'  --learning_rate ${LEARNING_RATE} 
EOT
    sbatch "${slurm_script_name}.sh"
}

if ! [ -d "runs" ]; then
  mkdir -p "runs"
fi
pushd runs

# declare -a BATCH_SIZE_ARR=("1000" "2000" )
declare -a TB_ARR=("200" "500")
declare -a NN_SIZE_ARR=("16" "64" "128")
declare -a NATSPI_ARR=("200" "500" "1000")
declare -a LR_ARR=("0.1" "0.01")

# for batch in "${BATCH_SIZE_ARR[@]}"
# do 
#     for tb in "${TB_ARR[@]}" 
#     do 
#         for natspi in "${NATSPI_ARR[@]}" 
#         do
#             for nnsize in "${NN_SIZE_ARR[@]}"
#             do 
#                 for lr in "${LR_ARR[@]}"
#                 do
#                     launch_mbrl_train_job $batch $tb $natspi $nnsize 2 $lr
#                 done
#             done
#         done
#     done
# done


for tb in "${TB_ARR[@]}" 
do 
    for natspi in "${NATSPI_ARR[@]}" 
    do
        for nnsize in "${NN_SIZE_ARR[@]}"
        do 
            for lr in "${LR_ARR[@]}"
            do
                launch_mbrl_train_job 1000 $tb $natspi $nnsize 2 $lr
            done
        done
    done
done
