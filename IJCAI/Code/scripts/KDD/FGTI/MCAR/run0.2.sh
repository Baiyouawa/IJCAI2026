#在运行脚本前，注意赋予执行权限: chmod +x run0.2.sh (Before running the script, make sure to give execute permission: chmod +x run0.2.sh)
SEEDS=(3407 3408 3409 3410 3411) # 5个随机种子，跑五次取平均 (We choose 5 different random seeds to run and average)
#注意，MCAR要修改的地方:self.mask = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/data/kdd_" (pay attention, for MCAR you need to modify:self.mask = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/data/kdd_")
for seed in "${SEEDS[@]}"
do
    echo " "
    echo "#############################################"
    echo "########## STARTING RUN WITH SEED: $seed ##########"
    echo "#############################################"
    echo " "
    python ../../../../A_diffusion_train.py --dataset kdd --missing_rate 0.2 --enc_in 99 --c_out 99 --seed "$seed" &
    PID=$!
    echo "--- Process started. PID: $PID ---"
    wait $PID
    echo "--- Process $PID finished. Now forcing kill to ensure VRAM clear... ---"
    kill -9 $PID 2>/dev/null
    sleep 3
done
echo " "
echo "#############################################"
echo "########## ALL 5 RUNS COMPLETED! ##########"
echo "#############################################"