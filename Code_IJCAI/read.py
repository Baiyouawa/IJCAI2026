import pandas as pd
import numpy as np

# --- 1. 定义你要读取的文件路径 ---
# 这必须和你上一个脚本的 output_path 完全一致
file_path = "/mnt/4T/IJCAI/FGTI/Data/Physio_norm.csv"

print(f"正在读取文件: {file_path}")

try:
    # --- 2. 加载数据 ---
    # 我们使用 pandas 来读取，因为它能更漂亮地打印行
    # header=None 是因为 np.savetxt 保存的文件没有列名
    data = pd.read_csv(file_path, header=None, delimiter=',')
    
    # --- 3. 打印维度 ---
    # .shape 会显示 (行数, 列数)
    print("--- 维度 (Shape) ---")
    print(f" (行数, 列数): {data.shape}")
    print("\n")

    # --- 4. 打印前 50 行 ---
    print("--- 文件的前 50 行 ---")
    # .head(50) 会抓取前 50 行并以表格形式显示
    print(data.head(50))


except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")