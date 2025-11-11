import numpy as np
from sklearn.preprocessing import StandardScaler
# from tsdb.data_processing import load_physionet2012 # 1. 不再需要这个导入
from sklearn.preprocessing import StandardScaler
import pandas as pd  # 1. (新) 导入 pandas 来手动读取文件
import os            # 1. (新) 导入 os 来处理文件路径
import glob          # 1. (新) 导入 glob 来查找文件

# --- 2. (新) 手动数据加载 ---
BASE_DIR = '/mnt/4T/IJCAI/FGTI/Data/phy'
print(f"Manually loading data from: {BASE_DIR}")

# (A) 加载所有标签 (Outcomes-*.txt)
outcome_files = glob.glob(os.path.join(BASE_DIR, 'Outcomes-*.txt'))
y_list = [pd.read_csv(f) for f in outcome_files]
y = pd.concat(y_list)
y = y.set_index('RecordID') # 将 RecordID 设置为索引，以便后续查找
print(f"Loaded {len(y)} labels (y).")


# --- (B) (重大更新) 加载所有特征 (set-a, set-b, set-c 里的 .txt) ---
all_patient_data = []
folders_to_scan = ['set-a', 'set-b', 'set-c']

# 定义所有可能的参数，这将成为我们的列
# (这是从数据集中获取的)
ALL_PARAMS = [
    'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT',
    'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP',
    'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat',
    'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
    'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]
# 'Time' and 'RecordID' will be added separately

for folder in folders_to_scan:
    current_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(current_path):
        continue
    
    print(f"Scanning folder: {folder}...")
    txt_files = glob.glob(os.path.join(current_path, '*.txt'))
    
    for file_path in txt_files:
        try:
            record_id = int(os.path.basename(file_path).split('.txt')[0])
            
            # 1. 创建一个 48xN 的空 DataFrame (N = 48)
            patient_df = pd.DataFrame(index=range(48), columns=ALL_PARAMS)
            patient_df['Time'] = range(48)
            patient_df['RecordID'] = record_id
            
            # 2. 读取文件并填充
            time_series_data = {} # 存储时间序列值
            
            with open(file_path, 'r') as f:
                # 跳过第一行 'RecordID,12345'
                f.readline()
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    
                    if len(parts) == 2:
                        # 这是静态数据 (e.g., 'Age,54')
                        param, value = parts
                        if param in ALL_PARAMS:
                            try:
                                patient_df[param] = float(value) # 广播到所有48行
                            except ValueError:
                                pass # 忽略无法转换的值 (比如 'na')
                                
                    elif len(parts) == 3:
                        # 这是时间序列数据 (e.g., '00:00,HR,80')
                        time_str, param, value = parts
                        if param in ALL_PARAMS:
                            try:
                                hour = int(time_str.split(':')[0])
                                if 0 <= hour < 48:
                                    # 存储，稍后填充 (因为一小时内可能有多个值)
                                    if (hour, param) not in time_series_data:
                                        time_series_data[(hour, param)] = []
                                    time_series_data[(hour, param)].append(float(value))
                            except (ValueError, IndexError):
                                pass # 忽略格式错误

            # 3. 处理时间序列数据 (取一小时内的平均值)
            for (hour, param), values in time_series_data.items():
                if values:
                    patient_df.loc[hour, param] = np.mean(values)

            # F检查是否有有效数据 (模仿之前的 WARNING)
            if patient_df.drop(columns=['Time', 'RecordID']).isnull().all().all():
                 print(f"[WARNING]: Ignore {record_id}, because its len==1 (or no data)")
                 continue

            all_patient_data.append(patient_df)
            
        except Exception as e:
            # 捕获 `int(os.path.basename...)` 的错误，比如 'index.html'
            print(f"Error processing file {file_path}: {e}")

if not all_patient_data:
    print("ERROR: No patient data files (.txt) were found in set-a, set-b, or set-c.")
    # 退出，或者让它在下一步自然地失败
    
X = pd.concat(all_patient_data, ignore_index=True)
# 重新排序列，确保 RecordID 和 Time 在前面
cols = ['RecordID', 'Time'] + ALL_PARAMS
X = X[cols]

print(f"Loaded {len(X)} total time-series rows (X).")
# --- 手动加载结束 ---


# --- 3. (旧) 你的原始代码，现在可以正常工作了 ---

num_samples = len(X['RecordID'].unique())
print(f"Found {num_samples} unique samples.")

# 这个循环只是提取标签，不影响 X 的处理
train_set_idx = []
test_set_idx = []
labels = []
for num in range(num_samples):
    data_chunk = X[num*48: (num+1)*48]
    
    if data_chunk.empty:
        continue
        
    dataid = data_chunk["RecordID"].iloc[0]

    # .loc[dataid] 现在可以工作，因为 y 已经用 RecordID 作为索引了
    label = y.loc[dataid]["In-hospital_death"] 
    
    labels.append(label)

print("Preprocessing data (dropping columns, scaling, filling NaN)...")
# X.drop 现在可以安全地工作
X = X.drop(['RecordID', 'Time'], axis = 1) 
# StandardScaler 现在只会收到数值型(float)和NaN列
X = StandardScaler().fit_transform(X.to_numpy()) 
X[np.isnan(X)] = -200 # 将缺失值替换为-200

# 指定保存的最终路径
output_path = "/mnt/4T/IJCAI/FGTI/Data/Physio_norm.csv"
print(f"Saving processed data to: {output_path}")
np.savetxt(output_path, X, fmt="%.6f", delimiter=",")

print("All done.")