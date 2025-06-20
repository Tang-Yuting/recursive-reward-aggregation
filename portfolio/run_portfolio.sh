#!/bin/bash

SEEDS=(${1:-4})
ADAPT_STATE=(${2:-"False"})
ADAPT_REWARD=(${3:-"False"})

for SEED in "${SEEDS[@]}"; do
    SESSION_NAME="PPO_Portfolio_ours_multi_env_seed${SEED}"
    LOG_DIR="baseline_mc/ours_multi_env/${SEED}"
    mkdir -p $LOG_DIR

    tmux has-session -t $SESSION_NAME 2>/dev/null

    if [ $? != 0 ]; then
        tmux new-session -d -s $SESSION_NAME
        tmux send-keys -t $SESSION_NAME "source ~/your_env/bin/activate" C-m

        tmux send-keys -t $SESSION_NAME "python runner_full_exp_fin_env.py --seed_start $SEED --adapt_state $ADAPT_STATE --adapt_reward $ADAPT_REWARD --result_dir $LOG_DIR  | tee ${LOG_DIR}/ppo_state${ADAPT_STATE}_reward${ADAPT_REWARD}_seed${SEED}.log" C-m
        echo "[INFO] Started training in tmux session: $SESSION_NAME"
    fi
    sleep 2
done

echo "All tmux sessions started. Use 'tmux ls' to see running sessions."



