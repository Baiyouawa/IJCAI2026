import torch
import numpy as np
import random

mixed_missing_ps = [0.2, 0.4, 0.6, 0.8]
seeds = [3407, 3408, 3409, 3410, 3411]

import random
import numpy as np

def get_MAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(np.isnan(org_data))] = 0 
    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0] 
    # 第7列为温度 (the 7th column is temperature)
    attribute_data = org_data[:, 6]
    index = np.argsort(attribute_data)
    rank = np.argsort(index) + 1 
    inverted_rank = (time_step_num + 1) - rank
    rank_sum = np.sum(inverted_rank) 
    probability = inverted_rank / rank_sum 
    
    missing_sum = 0

    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 2)
        if attr >= 6:
            attr += 1
        x = np.random.choice(range(time_step_num), p=probability.ravel())
        
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag


def get_MNAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(np.isnan(org_data))] = 0 

    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        attribute_data = org_data[:, attr]
        index = np.argsort(attribute_data)
        rank = np.argsort(index) + 1
        inverted_rank = (time_step_num + 1) - rank
        rank_sum = np.sum(inverted_rank)
        probability = inverted_rank / rank_sum
        x = np.random.choice(range(time_step_num), p=probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag

def get_MIXED_mask_flag(org_data, seed, missing_rate, mar_proportion=0.5):
    """
    生成混合了MAR和MNAR的掩码.
    :param org_data: 原始数据
    :param seed: 随机种子
    :param missing_rate: *总*缺失率 (例如 0.4)
    :param mar_proportion: MAR在总缺失中占的比例 (例如 0.5, 表示MAR和MNAR各占一半)
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. 初始化基础掩码 (处理已有的NaN)
    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(np.isnan(org_data))] = 0 
    
    total_observable = np.sum(mask_flag)
    
    # 2. 计算 MAR 和 MNAR 各自需要缺失的*绝对数量*
    mar_target_count = int(total_observable * missing_rate * mar_proportion)
    mnar_target_count = int(total_observable * missing_rate * (1.0 - mar_proportion))

    time_step_num = org_data.shape[0]

    # --- 3. 执行 MAR 缺失 ---
    # MAR的概率逻辑 (基于第7列温度)
    attribute_data_mar = org_data[:, 6]
    index_mar = np.argsort(attribute_data_mar)
    rank_mar = np.argsort(index_mar) + 1 
    inverted_rank_mar = (time_step_num + 1) - rank_mar
    rank_sum_mar = np.sum(inverted_rank_mar) 
    probability_mar = inverted_rank_mar / rank_sum_mar 
    
    missing_sum_mar = 0
    while missing_sum_mar <= mar_target_count:
        # MAR逻辑: 避开第6列(温度)和最后一列
        attr = random.randint(0, org_data.shape[1] - 2)
        if attr >= 6:
            attr += 1
        
        x = np.random.choice(range(time_step_num), p=probability_mar.ravel())
        
        if mask_flag[x, attr] == 0: # 如果已经缺失, 则跳过
            continue
        mask_flag[x, attr] = 0
        missing_sum_mar += 1

    # --- 4. 执行 MNAR 缺失 ---
    # 注意: 在 MAR 已经修改过的 mask_flag 上继续操作
    missing_sum_mnar = 0
    while missing_sum_mnar <= mnar_target_count:
        # MNAR逻辑: 随机选择一列
        attr = random.randint(0, org_data.shape[1] - 1)
        
        # MNAR概率逻辑 (基于所选列的自身数据)
        attribute_data_mnar = org_data[:, attr]
        index_mnar = np.argsort(attribute_data_mnar)
        rank_mnar = np.argsort(index_mnar) + 1
        inverted_rank_mnar = (time_step_num + 1) - rank_mnar
        rank_sum_mnar = np.sum(inverted_rank_mnar)
        
        # 避免除以零 (如果某列数据全相同)
        if rank_sum_mnar == 0:
            probability_mnar = np.ones_like(inverted_rank_mnar) / time_step_num
        else:
            probability_mnar = inverted_rank_mnar / rank_sum_mnar

        x = np.random.choice(range(time_step_num), p=probability_mnar.ravel())
        
        if mask_flag[x, attr] == 0: # 如果已经缺失 (被MAR或之前的MNAR步骤), 则跳过
            continue
        mask_flag[x, attr] = 0
        missing_sum_mnar += 1

    return mask_flag




for missing_p in mixed_missing_ps:
    for seed in seeds:
        print(f"正在生成: total_missing_rate={missing_p}, seed={seed}") 

        # 加载数据 (每次循环加载以重置)
        a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv", delimiter=",")

        # --- 混合掩码生成逻辑 ---
        
        # 调用新的混合函数, MAR和MNAR各占50% (0.5)
        mixed_mask = get_MIXED_mask_flag(org_data=a, 
                                         seed=seed, 
                                         missing_rate=missing_p, 
                                         mar_proportion=0.5)

        # 6. 保存新的混合掩码
        save_path = (
            "/mnt/4T/IJCAI/FGTI/Data/data/kddmixed_" # 新的文件名前缀
            + str(missing_p)
            + "_"
            + str(seed)
            + ".csv"
        )
        
        np.savetxt(
            save_path,
            mixed_mask,
            fmt="%d",
            delimiter=",",
        )

# 你也可以选择性地保留生成纯MAR/MNAR掩码的旧循环
# ... (你原来的主循环代码可以放在这里) ...


print("所有*混合*掩码文件生成完毕！")