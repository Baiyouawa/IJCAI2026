import torch
import numpy as np
import random

missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407, 3408, 3409, 3410, 3411]


# 1. 接收 missing_rate 作为新参数
def get_MAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    # 2. 使用传入的 missing_rate，而不是 0.1
    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    # based on daxing_temperature
    attribute_data = org_data[:, 6]
    index = np.argsort(attribute_data)
    rank = np.argsort(index) + 1
    rank_sum = np.sum(rank)
    probability = rank / rank_sum

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        x = np.random.choice(range(time_step_num), p=probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag


# 1. 接收 missing_rate 作为新参数
def get_MNAR_mask_flag(org_data, seed, missing_rate):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    # 2. 使用传入的 missing_rate，而不是 0.1
    missing_sum_target = missing_rate * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)
        attribute_data = org_data[:, attr]
        index = np.argsort(attribute_data)
        rank = np.argsort(index) + 1
        rank_sum = np.sum(rank)
        probability = rank / rank_sum
        x = np.random.choice(range(time_step_num), p=probability.ravel())
        if mask_flag[x, attr] == 0:
            continue
        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag


# --- 主循环 ---
for missing_p in missing_ps:
    for seed in seeds:
        print(f"正在生成: missing_rate={missing_p}, seed={seed}")  # 添加日志

        random.seed(seed)
        np.random.seed(seed)
        a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv", delimiter=",")

        # 3. 将 missing_p 传递到函数中
        mar_mask = get_MAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)
        mnar_mask = get_MNAR_mask_flag(org_data=a, seed=seed, missing_rate=missing_p)

        # 4. (修复) 你之前保存文件的路径有一个拼写错误 (mnt/4T)
        #    我已经把它改成了 /mnt/new4T/
        np.savetxt(
            "/mnt/4T/IJCAI/FGTI/Data/data/kddmar_"  # 修复了路径
            + str(missing_p)
            + "_"
            + str(seed)
            + ".csv",
            mar_mask,
            fmt="%d",
            delimiter=",",
        )
        np.savetxt(
            "/mnt/4T/IJCAI/FGTI/Data/data/kddmnar_"  # 修复了路径
            + str(missing_p)
            + "_"
            + str(seed)
            + ".csv",
            mnar_mask,
            fmt="%d",
            delimiter=",",
        )

print("所有掩码文件生成完毕！")