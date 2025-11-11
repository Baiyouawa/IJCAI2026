# 定义一个包含所有种子的数组
SEEDS=(3407 3408 3409 3410 3411)

# 遍历这个数组
for seed in "${SEEDS[@]}"
do
    # 打印一个清晰的“分隔符”
    echo " "
    echo "#############################################"
    echo "########## STARTING RUN WITH SEED: $seed ##########"
    echo "#############################################"
    echo " "
    
    # 1. 在“后台”(&)运行你的 Python 命令
    #    这样脚本会立即开始运行，但 shell 循环可以继续到下一步
    python A_diffusion_train.py --dataset guangzhou --missing_rate 0.4 --enc_in 214 --c_out 214 --seed "$seed" --epoch_diff 400 --learning_rate_diff 1e-3 --device cuda:1 --batch 16&
    
    # 2. 立即获取刚启动的那个进程的 PID
    #    '$!' 是一个特殊变量，代表“上一个在后台运行的进程ID”
    PID=$!
    echo "--- Process started. PID: $PID ---"

    # 3. 等待这个 PID 运行结束
    #    'wait' 命令会“阻塞”这个 for 循环
    #    直到 PID=$PID 的进程（即你的python脚本）自己退出或崩溃
    wait $PID
    
    # 4. 脚本运行结束（或崩溃）后，执行你的“强制杀死”命令
    echo "--- Process $PID finished. Now forcing kill to ensure VRAM clear... ---"
    
    # 5. 强制杀死该 PID
    #    这会确保 *万一* 进程卡在退出阶段或没有完全释放资源，
    #    操作系统也会强行回收它的所有资源（包括显存）
    #    '2>/dev/null' 的意思是“如果进程已经死了，不要打印 'No such process' 错误”
    kill -9 $PID 2>/dev/null
    
    # 6. (推荐) 等待几秒钟
    #    给操作系统和 CUDA 驱动一点时间来完成“清扫”工作
    sleep 3

done

# 跑完 5 次后，打印一个最终提示
echo " "
echo "#############################################"
echo "########## ALL 5 RUNS COMPLETED! ##########"
echo "#############################################"