#!/bin/bash
# (这个脚本是 run0.1.sh [cite: run0.8.sh] 的“TimesNet 公平对比”版本)
SEEDS=(3407 3408 3409 3410 3411) # 5个随机种子，跑五次取平均 (We choose 5 different random seeds to run and average) [cite: run0.8.sh]

# (注意: 您必须在 run_timesnet.py [cite: run_timesnet_user_v2.py] 中添加 TimesNet [cite: timesnet_model_user.py] 专用的参数)
# (例如 d_model, e_layers, d_ff, top_k 等)
# (我在 run_timesnet.py [cite: run_timesnet_user_v2.py] 中已经看到了它们，所以这里是匹配的)

for seed in "${SEEDS[@]}"
do
    echo " "
    echo "#############################################"
    echo "########## STARTING TimeNet RUN WITH SEED: $seed ##########"
    echo "#############################################"
    echo " "
    
    # --- [核心修改] ---
    # 1. 调用 run_timesnet.py [cite: run_timesnet_user_v2.py] (而不是 A_diffusion_train.py [cite: A_diffusion_train.py])
    # 2. 传入与 FGTI [cite: run0.8.sh] 完全相同的参数
    # 3. (run_timesnet.py [cite: run_timesnet_user_v2.py] 会自动读取它需要的额外参数, 如 d_model [cite: run_timesnet_user_v2.py])
    python ../../../../run_timesnet.py \
    --dataset kdd \
    --missing_rate 0.2 \
    --enc_in 99 \
    --c_out 99 \
    --seq_len 48 \
    --batch 16 \
    --epoch_diff 200 \
    --learning_rate_diff 1e-3 \
    --seed "$seed" \
    --task_name 'imputation' \
    --d_model 128 \
    --e_layers 2 \
    --d_ff 2048 \
    --top_k 5 \&
    # --- [修改结束] ---
    
    PID=$!
    echo "--- TimeNet Process started. PID: $PID ---"
    wait $PID
    echo "--- Process $PID finished. Now forcing kill to ensure VRAM clear... ---"
    kill -9 $PID 2>/dev/null
    sleep 3
done
echo " "
echo "#############################################"
echo "########## ALL 5 TimeNet RUNS COMPLETED! ##########"
