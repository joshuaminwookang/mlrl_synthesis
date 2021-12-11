#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1: Model Training Hyperparameters: batch size initial "
    python rl_synth/scripts/run.py --exp_name mbtrain_batch200_natspi5_1x16 --env_name synthesis-v0 \
        --add_sl_noise --n_iter 1 --train_batch_size 10\
        --batch_size_initial 200 --num_agent_train_steps_per_iter 5 \
        --n_layers 1 --size 16 --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'
    python rl_synth/scripts/run.py --exp_name mbtrain_batch500_natspi5_1x16 --env_name synthesis-v0 \
        --add_sl_noise --n_iter 1 --train_batch_size 10\
        --batch_size_initial 500 --num_agent_train_steps_per_iter 5 \
        --n_layers 1 --size 16 --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'
    python rl_synth/scripts/run.py --exp_name mbtrain_batch1000_natspi5_1x16 --env_name synthesis-v0 \
        --add_sl_noise --n_iter 1 --train_batch_size 10\
        --batch_size_initial 1000 --num_agent_train_steps_per_iter 5 \
        --n_layers 1 --size 16 --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'
    # python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 \
    #     --add_sl_noise --n_iter 1 \
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 \
    #     --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1 \
    #     --mpc_action_sampling_strategy 'random'
    # python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 \
    #     --add_sl_noise --n_iter 1 \
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 \
    #     --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 \
    #     --mpc_action_sampling_strategy 'random'
    # python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 \
    #     --add_sl_noise --n_iter 1 \
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 \
    #     --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 \
    #     --mpc_action_sampling_strategy 'random'
elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2: MBRL with Trained MPC Policy"
    python rl_synth/scripts/run.py --exp_name debug2 --env_name synthesis-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 2 --n_iter 1 \
    --batch_size_initial 100 --batch_size 50 --mpc_horizon 2 -tb 25\
    --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    # python rl_synth/scripts/run.py --exp_name debug2 --env_name synthesis-v0 \
    # --add_sl_noise --num_agent_train_steps_per_iter 2 --n_iter 1 \
    # --batch_size_initial 500 --batch_size 100 --mpc_horizon 3 -tb 50\
    # --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    # python cs285/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
    # --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
    # --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
    # --mpc_action_sampling_strategy 'random' --video_log_freq -1
elif [[ $EXP_NUM == 3 ]]; then
    echo "Experiment 3: MBRL with Trained MPC Policy--Longer"
    python cs285/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-cs285-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 12 \
    --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
    --mpc_action_sampling_strategy 'random' --video_log_freq -1 --seed 3
    python cs285/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-cs285-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 1000 --n_iter 15 \
    --batch_size_initial 5000 --batch_size 5000 --mpc_horizon 10 \
    --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name  q3_cheetah --env_name cheetah-cs285-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 1500 --n_iter 20 \
    --batch_size_initial 5000 --batch_size 5000 --mpc_horizon 15 \
    --mpc_action_sampling_strategy 'random' --video_log_freq -1 
elif [[ $EXP_NUM == 4 ]]; then
    echo "Experiment 4: MBRL with Trained MPC Policy Variables"
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon5 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon15 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon30 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq100 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --video_log_freq -1 \
    --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 --mpc_action_sampling_strategy 'random'
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq1000 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --video_log_freq -1 \
    --batch_size 800 --n_iter 15 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'random'
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble1 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --video_log_freq -1 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --ensemble_size 1 --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble3 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --ensemble_size 3 --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble5 --env_name reacher-cs285-v0 \
    --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 \
    --batch_size 800 --n_iter 15 --mpc_action_sampling_strategy 'random' --ensemble_size 5 --video_log_freq -1 
elif [[ $EXP_NUM == 5 ]]; then
    echo "Experiment 5: CEM on Cheetah"
    # python cs285/scripts/run_hw4_mb.py --exp_name debug --env_name obstacles-cs285-v0 \
    # --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 12 \
    # --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
    # --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --video_log_freq -1 --seed 3
    # python cs285/scripts/run_hw4_mb.py --exp_name debug --env_name 'cheetah-cs285-v0' --mpc_horizon 15 \
    #  --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 \
    #  --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --video_log_freq -1 
    # python cs285/scripts/run_hw4_mb.py --exp_name q5_cheetah_random --env_name 'cheetah-cs285-v0' --mpc_horizon 15 \
    #  --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 \
    #  --mpc_action_sampling_strategy 'random' --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q5_cheetah_cem_2 --env_name 'cheetah-cs285-v0' --mpc_horizon 15 \
     --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 \
     --mpc_action_sampling_strategy 'cem' --cem_iterations 2 --video_log_freq -1 
    python cs285/scripts/run_hw4_mb.py --exp_name q5_cheetah_cem_4 --env_name 'cheetah-cs285-v0' --mpc_horizon 15 \
     --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 \
     --mpc_action_sampling_strategy 'cem' --cem_iterations 4  --video_log_freq -1 
else
    echo "Nothing to be done"
fi


