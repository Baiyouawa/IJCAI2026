import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

# --- 1. 定义文件路径 ---
# 确保 tensor.mat 在这个路径
mat_file_path = "/mnt/4T/IJCAI/FGTI/Data/tensor.mat" 
# 这是我们最终要输出的、正确的文件
output_csv_path = "/mnt/4T/IJCAI/FGTI/Data/Guangzhou_norm.csv" 

print(f"开始处理: {mat_file_path}")

try:
    # --- 2. 加载 .mat 文件并重塑 ---
    mat_data = sio.loadmat(mat_file_path)
    # 使用 'tensor' 键, 形状 (214, 61, 144)
    tensor_3d = mat_data['tensor']  
    
    # 重塑为 2D: (路段, 天, 窗口) -> (路段, 总时间步) -> (总时间步, 路段)
    # (214, 61*144) -> (214, 8784) -> (8784, 214)
    data_2d = tensor_3d.reshape(tensor_3d.shape[0], -1).T
    
    print(f"数据已重塑为 2D 形状: {data_2d.shape}") # 必须是 (8784, 214)

    # --- 3. 正确处理缺失值 (0) 并进行标准化 ---
    
    # 复制一份数据，确保是 float 类型
    data_to_norm = data_2d.copy().astype(float) 

    # 关键: 将所有 0 (原始缺失值) 临时替换为 np.nan
    # 这样 StandardScaler 在计算均值和方差时会*自动忽略*它们
    missing_mask = (data_to_norm == 0)
    data_to_norm[missing_mask] = np.nan
    
    print("已将 0 标记为 np.nan，准备进行逐列标准化...")

    # 初始化 StandardScaler
    scaler = StandardScaler()

    # 对 214 列分别进行 fit_transform (逐列标准化)
    data_normalized = scaler.fit_transform(data_to_norm)
    
    print("数据已完成逐列标准化。")

    # --- 4. 恢复缺失值标记 ---
    # StandardScaler 会把 np.nan 的位置输出为 np.nan
    # 我们将它们统一设置回 -200 (你的下游代码期望的标记)
    nan_mask = np.isnan(data_normalized)
    data_normalized[nan_mask] = -200
    
    print(f"已将所有缺失值统一标记为 -200。")

    # --- 5. 保存新的 .csv 文件 ---
    np.savetxt(
        output_csv_path, 
        data_normalized, 
        delimiter=",", 
        fmt="%.6f" # 保留6位小数
    )
    
    print(f"--- 成功! ---")
    print(f"已将处理好的 (8784, 214) 数据保存到: {output_csv_path}")


except Exception as e:
    print(f"处理过程中出错: {e}")