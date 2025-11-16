# 加载KDD气象数据集并进行标准化处理 (Load KDD meteorological dataset and perform normalization)
import pandas as pd
import numpy as np

#指定文件数据是逗号分隔，header=0表示第一行是列名，读取到的DataFrame转换为numpy数组 (Specify that the file data is comma-separated, header=0 means the first row is the column name, and convert the read DataFrame to a numpy array)
data = pd.read_csv("/mnt/4T/IJCAI/FGTI/Data/KDD.csv",delimiter=",", header=0).to_numpy()

list = []
for i in range(9):
    #选取所有行，去除所有Station ID和时间戳 (Select all rows, remove all Station IDs and timestamps)
    station_record = data[:, i*13+2: (i+1)*13]
    
    #将9个N行11列存入列表中(Store 9 N-row 11-column into the list)
    list.append(station_record) 

#原本是N*9*11的三维数据，变为N*99的二维数据 (Originally N*9*11 three-dimensional data, changed to N*99 two-dimensional data)
data = np.stack(list, axis=1).reshape(data.shape[0], -1)


means, stds = [], []
for j in range(data.shape[1]):
    data_j = []
    for i in range(data.shape[0]):
        #KDD数据集中原始缺失为空值，Pandas读取后变为NaN (In the KDD dataset, the original missing values are empty, and Pandas reads them as NaN)
        if np.isnan(data[i,j]):
            continue
        data_j.append(data[i,j])
    data_j = np.array(data_j)
    mean_j = np.mean(data_j)
    std_j = np.std(data_j)

    for i in range(data.shape[0]):
        if np.isnan(data[i,j]):
            continue
        data[i,j] = (data[i,j] - mean_j) / std_j
    
    means.append(mean_j)
    stds.append(std_j)

#将标准化后的数据保存为新的CSV文件 (Save the normalized data as a new CSV file)
np.savetxt("/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv",data, delimiter=",",fmt="%6f")