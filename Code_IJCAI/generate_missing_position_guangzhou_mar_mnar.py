import torch
import numpy as np
import random
import os # 导入 os 模块来创建文件夹

# 1. [修改] 将 missing_p 从单个值改为列表
missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407, 3408, 3409, 3410, 3411]

# 2. [修改] 函数现在接收 'missing_rate' (缺失率) 作为参数
def get_MAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    # 3. [修改] 使用传入的 'missing_rate'，而不是硬编码的 0.1
    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]
    
    # based on first sensor
    attribute_data = org_data[:, 0]
    index = np.argsort(attribute_data) 
    rank = np.argsort(index) + 1 

    # 4. --- [修改] 逻辑反转 ---
    # 原逻辑: probability = rank / rank_sum (值越高, 概率越高)
    # 新逻辑: 值越低 (rank=1), 概率越高
    # 我们反转 rank：rank=1(最低值) 变成 N, rank=N(最高值) 变成 1
    reversed_rank = time_step_num - rank + 1
    rank_sum = np.sum(reversed_rank)
    probability = reversed_rank / rank_sum
    # --- 逻辑反转结束 ---
    
    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        x = np.random.choice(range(time_step_num), p = probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1
    
    return mask_flag

# 2. [修改] 函数现在接收 'missing_rate' (缺失率) 作为参数
def get_MNAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    # 3. [修改] 使用传入的 'missing_rate'，而不是硬编码的 0.1
    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        attribute_data = org_data[:,attr]
        index = np.argsort(attribute_data) 
        rank = np.argsort(index) + 1 
        
        # 4. --- [修改] 逻辑反转 ---
        # 原逻辑: probability = rank / rank_sum (值越高, 概率越高)
        # 新逻辑: 值越低 (rank=1), 概率越高
        reversed_rank = time_step_num - rank + 1
        rank_sum = np.sum(reversed_rank)
        probability = reversed_rank / rank_sum
        # --- 逻辑反转结束 ---

        x = np.random.choice(range(time_step_num), p = probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag

# --- 主循环 ---

# [新增] 自动创建保存掩码的文件夹，防止报错
output_dir = "/mnt/4T/IJCAI/FGTI/Data/data"
os.makedirs(output_dir, exist_ok=True)
print(f"确保掩码文件夹存在: {output_dir}")

# 5. [修改] 添加了外层循环来遍历 missing_ps
for missing_p in missing_ps:
    for seed in seeds:
        print(f"正在生成: missing_rate={missing_p}, seed={seed}") # 添加日志
        
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/Guangzhou_norm.csv", delimiter=",")
        except FileNotFoundError:
            print("错误: 找不到 'Data/Guangzhou_norm.csv'")
            print("请确保你在正确的目录下运行此脚本，并且数据文件存在。")
            break # 退出循环
            
        # 6. [修改] 将 missing_p 传递给函数
        mar_mask = get_MAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)
        mnar_mask = get_MNAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)

        # 7. 文件名现在会根据 'missing_p' 循环正确地命名
        np.savetxt(f"{output_dir}/guangzhoumar_{missing_p}_{seed}.csv", mar_mask, fmt="%d", delimiter=",")
        np.savetxt(f"{output_dir}/guangzhoumnar_{missing_p}_{seed}.csv", mnar_mask, fmt="%d", delimiter=",")

print("所有40个掩码文件生成完毕！")