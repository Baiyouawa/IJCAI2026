#生成KDD的MCAR缺失文件 (Generate MCAR missing position file for KDD)
import torch
import numpy as np
import random
missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407,3408,3409,3410,3411]

for missing_p in missing_ps:
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        a = np.loadtxt("/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv", delimiter=",")

        mask_org = np.ones_like(a)
        mask_org[np.where(np.isnan(a))] = 0

        x = a.shape[0]
        y = a.shape[1]

        mask_target = mask_org.copy()

        missing_sum = 0
        missing_target_sum = np.sum(mask_org) * missing_p
        while missing_sum <= missing_target_sum:
            #随机在 [0, x-1] 范围内选一个整数作为行索引 i (Randomly select an integer in the range [0, x-1] as the row index i)
            i = random.randint(0, x-1)
            j = random.randint(0, y-1)    
            if mask_target[i,j] == 0:
                continue
            mask_target[i,j] = 0
            missing_sum += 1

        np.savetxt("/mnt/4T/IJCAI/FGTI/Data/data/kdd_" + str(missing_p) + "_" + str(seed) + ".csv", mask_target, fmt="%d", delimiter=",")