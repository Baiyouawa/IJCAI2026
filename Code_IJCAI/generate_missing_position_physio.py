import torch
import numpy as np
import random

# 定义要生成的缺失率和随机种子列表
missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407, 3408, 3409, 3410, 3411]

# 循环遍历每一种缺失率
for missing_p in missing_ps:
    # 循环遍历每一个随机种子
    for seed in seeds:
        print(f"Generating mask for missing_p={missing_p}, seed={seed}...")

        # 设置随机种子以保证结果可复现
        random.seed(seed)
        np.random.seed(seed)

        # 加载数据
        # 注意：请确保文件路径正确
        try:
            a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/Physio_norm.csv", delimiter=",")
        except IOError as e:
            print(f"Error loading file: {e}")
            print("Skipping this iteration.")
            continue

        # 创建原始掩码 (mask_org)
        # 1 表示存在数据, 0 表示缺失数据 (原始缺失值为 -200)
        mask_org = np.ones_like(a)
        mask_org[np.where(a == -200)] = 0

        # 获取数据维度
        x = a.shape[0]
        y = a.shape[1]
        
        # 打印原始数据的存在率
        # org_ratio = np.sum(mask_org) / (x * y)
        # print(f"  Original data presence ratio: {org_ratio:.4f}")

        # 复制原始掩码作为目标掩码的起点
        mask_target = mask_org.copy()

        # ------------------------------------------------------------------
        # 模仿第一个脚本的核心逻辑：
        # 1. 计算*已存在*数据的数量
        num_observed = np.sum(mask_org)
        
        # 2. 计算需要*新增*的缺失值的*目标数量*
        # (即，按已存在数据的 missing_p 比例来使其缺失)
        missing_target_sum = int(num_observed * missing_p)
        
        # 3. 初始化已新增的缺失值计数器
        missing_sum = 0
        
        # 循环直到新增的缺失值达到目标数量
        while missing_sum < missing_target_sum:
            # 随机选择一个位置
            i = random.randint(0, x - 1)
            j = random.randint(0, y - 1)

            # 检查这个位置是否*已经*是缺失的 (无论是原始缺失还是新引入的)
            if mask_target[i, j] == 0:
                continue  # 如果是 0 (已缺失)，则跳过，重新选择
            
            # 如果不是 0 (即 mask_target[i, j] == 1)，则将其设置为缺失
            mask_target[i, j] = 0
            
            # 将新增缺失计数器加 1
            missing_sum += 1
        # ------------------------------------------------------------------

        # 打印最终掩码的存在率
        final_ratio = np.sum(mask_target) / (x * y)
        print(f"  Final data presence ratio: {final_ratio:.4f}")

        # 构建保存文件的路径
        save_path = f"/mnt/4T/IJCAI/FGTI/Data/data/physio_{missing_p}_{seed}.csv"
        
        # 保存新的掩码文件
        try:
            np.savetxt(save_path, mask_target, fmt="%d", delimiter=",")
            print(f"  Mask saved to {save_path}")
        except IOError as e:
            print(f"Error saving file: {e}")

print("All masks generated.")