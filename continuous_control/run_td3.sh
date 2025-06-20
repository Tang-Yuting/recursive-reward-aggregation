#!/bin/bash

SEEDS=(${1:-42})
ENV_NAME=${2:-"Ant-v3"}
FILE_ENV_NAME=${3:-"Ant"}
recursive_type=${4:-"dsum"}
output_number=${5:-1}


for SEED in "${SEEDS[@]}"; do
    SESSION_NAME="td3_${ENV_NAME}_${recursive_type}_seed${SEED}"
    LOG_DIR="result_TD3/${FILE_ENV_NAME}/${recursive_type}/${SEED}"
    mkdir -p $LOG_DIR
    tmux has-session -t $SESSION_NAME 2>/dev/null

    if [ $? != 0 ]; then
        tmux new-session -d -s $SESSION_NAME
        tmux send-keys -t $SESSION_NAME "source ~/your_env/bin/activate" C-m
        tmux send-keys -t $SESSION_NAME "python main_td3_${FILE_ENV_NAME}.py --seed $SEED --env $ENV_NAME --recursive_type $recursive_type --output_number $output_number --env_name $FILE_ENV_NAME | tee ${LOG_DIR}/td3_${ENV_NAME}_seed${SEED}.log" C-m
        echo "[INFO] Started training in tmux session: $SESSION_NAME"
    fi
    sleep 2
done

echo "All tmux sessions started. Use 'tmux ls' to see running sessions."
echo "Attach to a session using 'tmux attach -t ppo_<env>_seed<seed>'"

