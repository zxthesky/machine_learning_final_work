import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
data = pd.read_csv('./exp/final_homework/ETT-small/ETTh1.csv')

# 处理时间戳
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# 差分转换
def difference(dataset, interval=1):
    return dataset.diff(periods=interval)

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))

# 划分数据集
N = len(data)
train_data = data.iloc[:int(0.6 * N)]
val_data = data.iloc[int(0.6 * N):int(0.8 * N)]
test_data = data.iloc[int(0.8 * N):]

# 应用差分转换
train_diff = difference(train_data).dropna()
val_diff = difference(val_data).dropna()
test_diff = difference(test_data).dropna()

# 应用归一化
train_scaled = scaler.fit_transform(train_diff)
val_scaled = scaler.transform(val_diff)
test_scaled = scaler.transform(test_diff)

# 定义滑动窗口函数
def sliding_window(true_data ,data, input_window, output_window=1):  # 默认输出窗口为1
    X, y, x_true, y_true = [], [], [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:(i + input_window)])
        y.append(data[i + input_window: i + input_window + output_window])
        x_true.append(true_data[i:(i + input_window)])
        y_true.append(true_data[i + input_window: i + input_window + output_window])
    return np.array(X), np.array(y),np.array(x_true), np.array(y_true)

# 应用滑动窗口
input_window = 96
output_window = 1

X_train, y_train, x_train_true, y_train_true = sliding_window(train_data,train_scaled, input_window, output_window)
# 验证集和测试集仍然预测未来96小时
output_window = 96

X_val, y_val, x_val_true, y_val_true = sliding_window(val_data,val_scaled, input_window, output_window)
X_test, y_test, x_test_true, y_test_true = sliding_window(test_data,test_scaled, input_window, output_window)

# 转换为 PyTorch 张量
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
x_train_true = torch.Tensor(x_train_true)
x_val_true = torch.Tensor(x_val_true)
x_test_true = torch.Tensor(x_test_true)
y_train_true = torch.Tensor(y_train_true)
y_val_true = torch.Tensor(y_val_true)
y_test_true = torch.Tensor(y_test_true)
# 创建数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, x_true, y_true):
        self.X = X
        self.y = y
        self.y_true = y_true
        self.x_true = x_true

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_true[idx], self.x_true[idx]

# 创建 DataLoader
train_loader = DataLoader(TimeSeriesDataset(X_train, y_train,x_train_true, y_train_true), batch_size=64, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val,x_val_true , y_val_true), batch_size=64)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test,x_test_true, y_test_true), batch_size=64)

def inverse_scale(scaler, scaled_data):
    """逆归一化"""
    return scaler.inverse_transform(scaled_data)

def inverse_difference(original_data, predicted_diff):
    """逆差分"""
    # 初始化逆差分结果数组，初始值为 original_data
    inverted = np.zeros_like(predicted_diff)
    inverted[0] = original_data + predicted_diff[0]

    # 累加差分值
    for i in range(1, len(predicted_diff)):
        inverted[i] = inverted[i - 1] + predicted_diff[i]

    return inverted

class LSTMSeq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMSeq2SeqModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出维度为7，对应7个特征的预测

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # LSTM层
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 只使用最后一个时间步的输出来进行预测
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_out)

        return prediction.unsqueeze(1)

# 模型参数
input_dim = 7    # 输入特征的数量
hidden_dim = 128  # LSTM 隐藏层的大小
num_layers = 4   # LSTM 层数
output_dim = 7   # 输出特征的数量
# 创建模型
model = LSTMSeq2SeqModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
# 训练模型
def recursive_val_predict(model, input_sequence, future_steps):
    model.eval()
    predictions = []
    last_sequence = input_sequence.clone()

    for _ in range(future_steps):
        with torch.no_grad():
            prediction = model(last_sequence)
            predictions.append(prediction)
            # 更新输入序列以包括新的预测
            last_sequence = torch.cat((last_sequence[:, 1:, :], prediction), dim=1)
    return torch.cat(predictions, dim=1)


def train_and_evaluate_model(model, train_loader, val_loader, epochs=100, future_steps=96, model_save_path='./exp/final_homework/best_model.pth'):
    if os.path.exists(model_save_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
    else:
        best_val_mse = float('inf')
        epochs_no_improve = 0
        patience = 10  # 早停的耐心参数
        for epoch in range(epochs):
            # 训练过程
            model.train()
            total_train_loss = 0
            for X_batch, y_batch,_,_ in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                train_loss = criterion(outputs, y_batch[:, -1, :])
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}')
            # 验证过程
            model.eval()
            total_val_mse = 0
            total_val_mae = 0
            with torch.no_grad():
                for X_batch, y_batch,_,_ in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    # 递归预测
                    val_outputs = recursive_val_predict(model, X_batch, future_steps)
                    # import pdb;pdb.set_trace()
                    val_mse = F.mse_loss(val_outputs, y_batch)
                    val_mae = F.l1_loss(val_outputs, y_batch)
                    total_val_mse += val_mse.item() * X_batch.size(0)
                    total_val_mae += val_mae.item() * X_batch.size(0)

            avg_val_mse = total_val_mse / len(val_loader.dataset)
            avg_val_mae = total_val_mae / len(val_loader.dataset)
            print(f'Epoch {epoch}, Val MSE: {avg_val_mse}, Val MAE: {avg_val_mae}')
            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                epochs_no_improve = 0
                print(f"Epoch {epoch}, improved Val MSE: {avg_val_mse}, saving model...")
                torch.save(model.state_dict(), model_save_path)  # 保存模型
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break

# 创建并编译模型

# 训练模型
train_and_evaluate_model(model, train_loader, val_loader, epochs=1000)

def evaluate_on_test_set(model, test_loader, scaler, future_steps=96):
    model.eval()
    total_test_mse = 0
    total_test_mae = 0
    plot_data = None
    with torch.no_grad():
        for batch_index, (X_batch, _, x_true_batch , y_true_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            x_true_batch = x_true_batch.numpy()  # 获取原始输入序列的真实值
            y_true_batch = y_true_batch.numpy()  # 获取原始输出序列的真实值
            test_outputs = recursive_val_predict(model, X_batch, future_steps)  # (batch_size, 96, 7)

            # 逐个处理批次中的预测结果
            for i in range(test_outputs.shape[0]):
                # 逆归一化
                batch_output_inverse = inverse_scale(scaler, test_outputs[i].cpu().numpy())  # (96, 7)

                # 逆差分
                last_original_value = y_true_batch[i, -1]  # 使用该批次的最后一个原始真实值
                batch_output_original = inverse_difference(last_original_value, batch_output_inverse)  # (96, 7)

                # 真实值
                y_true = y_true_batch[i]  # (96, 7)

                # 计算 MSE 和 MAE
                mse = mean_squared_error(y_true, batch_output_original)
                mae = mean_absolute_error(y_true, batch_output_original)
                total_test_mse += mse
                total_test_mae += mae
                if batch_index == 5 and i == 5:  # 选择第一个批次的第一个数据点
                    # 选择一个特征进行绘图，这里选择第一个特征
                    feature_index = 6
                    original_input_sequence = x_true_batch[i, :, feature_index]  # 原始输入序列
                    original_output_sequence = y_true_batch[i, :, feature_index]  # 原始输出序列
                    predicted_sequence = batch_output_original[:, feature_index]

                    # 组合输入序列和真实值序列
                    full_original_sequence = np.concatenate((original_input_sequence, original_output_sequence))

                    plot_data = (full_original_sequence, predicted_sequence)
                    import pdb;pdb.set_trace()
    avg_test_mse = total_test_mse / len(test_loader)
    avg_test_mae = total_test_mae / len(test_loader)
    if plot_data:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data[0], label='Original Sequence')
        plt.plot(range(len(plot_data[0]) - len(plot_data[1]), len(plot_data[0])), plot_data[1], label='Predicted Sequence')
        plt.legend()
        plt.title('Comparison of Original and Predicted Sequences')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.show()
    return avg_test_mse, avg_test_mae

# 对测试集进行评估
avg_test_mse, avg_test_mae = evaluate_on_test_set(model, test_loader, scaler)
print(f"Test MSE: {avg_test_mse}, Test MAE: {avg_test_mae}")
