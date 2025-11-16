#预览读取文件(Preview reading file)
import pandas as pd
import numpy as np

file_path = "/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv"
#我们的数据存放在移动硬盘中的Data文件夹中 (our data is stored in the Data folder of the external hard drive)
print(f"正在读取文件: {file_path}")
try:
    data = pd.read_csv(file_path, header=None, delimiter=',')
    
    print("--- 维度 (Shape) ---")
    print(f" (行数, 列数): {data.shape}")
    print("\n")

    print("--- 文件的前 50 行 ---")
    print(data.head(50))
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")