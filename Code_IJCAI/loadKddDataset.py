import pandas as pd
import numpy as np
#分割符为逗号，且第一行为表头
data = pd.read_csv("/mnt/4T/IJCAI/FGTI/Data/KDD.csv",delimiter=",", header=0).to_numpy()

list = []
for i in range(9):
    #remove stationId and utc_time(去除掉ID和Time)
    station_record = data[:, i*13+2: (i+1)*13]
    list.append(station_record) #变成N*99
data = np.stack(list, axis=1).reshape(data.shape[0], -1)
#N*9*11变为N*99
means, stds = [], []
for j in range(data.shape[1]):
    data_j = []
    for i in range(data.shape[0]):
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
    #计算均值方差---标准化
np.savetxt("/mnt/4T/IJCAI/FGTI/Data/KDD_norm.csv",data, delimiter=",",fmt="%6f")