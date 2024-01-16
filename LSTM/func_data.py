import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 加载数据
data = pd.read_csv('./exp/final_homework/ETT-small/ETTh1.csv')

# 处理时间戳
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# 划分数据集
N = len(data)
train_data = data.iloc[:int(0.6 * N)]
val_data = data.iloc[int(0.6 * N):int(0.8 * N)]
test_data = data.iloc[int(0.8 * N):]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# 定义滑动窗口函数
def sliding_window(data, input_window, output_window):
    X, y_96, y_336 = [], [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:(i + input_window)])
        y_96.append(data[(i + input_window):(i + input_window + 96)])
        y_336.append(data[(i + input_window):(i + input_window + 336)])
    return np.array(X), np.array(y_96), np.array(y_336)

# 应用滑动窗口
input_window = 96
output_window_96 = 96
output_window_336 = 336

X_train, y_train_96, y_train_336 = sliding_window(train_data_scaled, input_window, max(output_window_96, output_window_336))
X_val, y_val_96, y_val_336 = sliding_window(val_data_scaled, input_window, max(output_window_96, output_window_336))
X_test, y_test_96, y_test_336 = sliding_window(test_data_scaled, input_window, max(output_window_96, output_window_336))

# 调整形状以适应模型的输入和输出
y_train_96 = y_train_96.reshape(-1, 96, data.shape[1])
y_train_336 = y_train_336.reshape(-1, 336, data.shape[1])
y_val_96 = y_val_96.reshape(-1, 96, data.shape[1])
y_val_336 = y_val_336.reshape(-1, 336, data.shape[1])
y_test_96 = y_test_96.reshape(-1, 96, data.shape[1])
y_test_336 = y_test_336.reshape(-1, 336, data.shape[1])

# 保存数据
np.save('./exp/final_homework/ETTh1/X_train.npy', X_train)
np.save('./exp/final_homework/ETTh1/y_train_96.npy', y_train_96)
np.save('./exp/final_homework/ETTh1/y_train_336.npy', y_train_336)
np.save('./exp/final_homework/ETTh1/X_val.npy', X_val)
np.save('./exp/final_homework/ETTh1/y_val_96.npy', y_val_96)
np.save('./exp/final_homework/ETTh1/y_val_336.npy', y_val_336)
np.save('./exp/final_homework/ETTh1/X_test.npy', X_test)
np.save('./exp/final_homework/ETTh1/y_test_96.npy', y_test_96)
np.save('./exp/final_homework/ETTh1/y_test_336.npy', y_test_336)
