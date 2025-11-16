import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

mat_file_path = "/mnt/4T/IJCAI/FGTI/Data/tensor.mat" 
output_csv_path = "/mnt/4T/IJCAI/FGTI/Data/Guangzhou_norm.csv" 

print(f"开始处理: {mat_file_path}")

try:
    mat_data = sio.loadmat(mat_file_path)
    tensor_3d = mat_data['tensor']  
    
    #重塑行为时间，列为路口数 (Reshape rows as time, columns as intersections)
    data_2d = tensor_3d.reshape(tensor_3d.shape[0], -1).T

    # 8784行，214列 (8784 rows, 214 columns)
    print(f"数据已重塑为 2D 形状: {data_2d.shape}") 

    data_to_norm = data_2d.copy().astype(float) 

    #在广州数据集中，原始缺失就显示为0 (In the Guangzhou dataset, original missing values are represented as 0)
    missing_mask = (data_to_norm == 0)
    data_to_norm[missing_mask] = np.nan
    
    print("已将 0 标记为 np.nan，准备进行逐列标准化...")

    scaler = StandardScaler()

    data_normalized = scaler.fit_transform(data_to_norm)
    
    print("数据已完成逐列标准化。")

    # 我们将它们统一设置回 -200 作为缺失值标记 (We will uniformly set them back to -200 as missing value markers)

    nan_mask = np.isnan(data_normalized)
    data_normalized[nan_mask] = -200
    
    print(f"已将所有缺失值统一标记为 -200。")

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