import torch
import numpy as np
import random
import os 

missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407, 3408, 3409, 3410, 3411]

def get_MAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]
    #这里选取第一个路口的流量作为影响属性 (the first column is traffic flow of the first intersection)
    attribute_data = org_data[:, 0]
    index = np.argsort(attribute_data) 
    rank = np.argsort(index) + 1 
    reversed_rank = time_step_num - rank + 1
    rank_sum = np.sum(reversed_rank)
    probability = reversed_rank / rank_sum

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(1, org_data.shape[1] - 1)
        x = np.random.choice(range(time_step_num), p = probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1
    
    return mask_flag

def get_MNAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        attribute_data = org_data[:,attr]
        index = np.argsort(attribute_data) 
        rank = np.argsort(index) + 1 
        
        reversed_rank = time_step_num - rank + 1
        rank_sum = np.sum(reversed_rank)
        probability = reversed_rank / rank_sum

        x = np.random.choice(range(time_step_num), p = probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag

output_dir = "/mnt/4T/IJCAI/FGTI/Data/data"
os.makedirs(output_dir, exist_ok=True)
print(f"确保掩码文件夹存在: {output_dir}")

for missing_p in missing_ps:
    for seed in seeds:
        print(f"正在生成: missing_rate={missing_p}, seed={seed}") 
        
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/Guangzhou_norm.csv", delimiter=",")
        except FileNotFoundError:
            print("错误: 找不到 'Data/Guangzhou_norm.csv'")
            print("请确保你在正确的目录下运行此脚本，并且数据文件存在。")
            break 

        mar_mask = get_MAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)
        mnar_mask = get_MNAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)

        np.savetxt(f"{output_dir}/guangzhoumar_{missing_p}_{seed}.csv", mar_mask, fmt="%d", delimiter=",")
        np.savetxt(f"{output_dir}/guangzhoumnar_{missing_p}_{seed}.csv", mnar_mask, fmt="%d", delimiter=",")

print("所有40个掩码文件生成完毕！")