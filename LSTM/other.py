import argparse
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# 随机数种子
seed = 9
np.random.seed(seed)

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def plot_loss_data(results_loss, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(results_loss, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')  # 保存到当前目录，你可以修改路径
    plt.show()

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, device):
        self.sequences = sequences
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence).to(self.device), torch.Tensor(label).to(self.device)

def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def calculate_mse(predictions, labels):
    return ((predictions - labels) ** 2).mean()

def create_dataloader(config, device):
    df = pd.read_csv(config.data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口
    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)
    cols_data = df.columns[1:]
    df_data = df[cols_data]
    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values
    # 定义标准化优化器
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)
    train_data = true_data[int(0.4 * len(true_data)):]
    valid_data = true_data[int(0.2 * len(true_data)):int(0.4 * len(true_data))]
    test_data = true_data[:int(0.2* len(true_data))]
    # print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))
    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)
    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)
    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq, device)
    test_dataset = TimeSeriesDataset(test_inout_seq, device)
    valid_dataset = TimeSeriesDataset(valid_inout_seq, device)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    # print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    # print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    return train_loader, test_loader, valid_loader, scaler

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, pre_len, hidden_size, n_layers,  dropout=0.05):
        super(LSTM, self).__init__()
        self.pre_len = pre_len
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, bias=True, batch_first=True)  
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size, obs_len, features_size = x.shape  
        xconcat = self.hidden(x)  
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(device) 
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            device)  
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)  
            out, (ht, ct) = self.lstm(xt, (ht, ct))  
            htt = ht[-1, :, :]  
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H) 
        x = self.linear(H)
        return x[:, -self.pre_len:, :]

def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    val_losses = []  # 记录验证集损失
    # 早停设置
    early_stopping_patience = 10
    min_val_loss = float('inf')
    patience_counter = 0

    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        avg_loss = sum(losss) / len(losss)
        tqdm.write(f"\t train: Epoch {i + 1} / {epochs}, Loss: {avg_loss}")
        results_loss.append(avg_loss)

        # 验证集损失计算
        val_loss = valid(model, valid_loader, device)
        tqdm.write(f"\t val:Epoch {i + 1} / {epochs}, Loss: {val_loss}")
        val_losses.append(val_loss)  # 添加验证集损失

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'save_model_{seed}.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"早停: 在第 {i + 1} 个epoch停止")
                break

    print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
    plot_loss_data(results_loss, val_losses)

def valid(model, valid_loader, device):
    model.eval()  # 设置为评估模式
    losss = []
    with torch.no_grad():  # 在评估模式下关闭梯度计算
        for seq, labels in valid_loader:
            seq, labels = seq.to(device), labels.to(device)
            pred = model(seq)
            mae = calculate_mae(pred.detach().cpu().numpy(), labels.cpu().numpy())
            losss.append(mae)
    model.train()  # 切换回训练模式
    return sum(losss) / len(losss)


def test(model, args, test_loader, scaler):
    model.load_state_dict(torch.load(f'./save_model_{seed}.pth'))
    model.eval()  # 评估模式
    mae_loss = []
    mse_loss = []  # 新增一个列表来存储MSE误差
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(), label.detach().cpu().numpy())
        mae_loss.append(mae)
        mse = calculate_mse(pred.detach().cpu().numpy(), label.detach().cpu().numpy())
        mse_loss.append(mse)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])
    print("测试集误差MAE:", sum(mae_loss) / len(mae_loss) if mae_loss else 0)
    print("测试集误差MSE:", sum(mse_loss) / len(mse_loss) if mse_loss else 0)
    # 假设labels和results的长度足够
    new_labels = labels[-432:]  # 获取最后192个数据点
    new_results = results[-336:]  # 获取最后96个数据点

    # 生成x坐标
    x_labels = range(len(new_labels))  # 对于labels, x坐标从0到191
    x_results = range(len(new_labels) - 336, len(new_labels))  # 对于results, x坐标从96到191

    # 绘制图形
    plt.plot(x_labels, new_labels, label='TrueValue')
    plt.plot(x_results, new_results, label='Prediction')
    plt.title("Test State")
    plt.legend()
    plt.show()



def rolling_predict(model, scaler, df, device, input_size=96, predict_size=96):
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式
    # 提取前192个数据点
    data = df.iloc[:input_size + predict_size, 1:].values  # 假设df的第一列不是特征列

    # 数据标准化
    data_scaled = scaler.transform(data)
    
    # 使用前96个数据点作为初始输入
    input_seq = data_scaled[:input_size]

    predictions = []
    for _ in range(predict_size):
        # 将输入转换为模型需要的格式
        input_tensor = torch.FloatTensor(input_seq[-input_size:]).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            pred = model(input_tensor)
        pred = pred.detach().cpu().numpy()[0, 0, :]  # 假设输出是(batch_size, 1, n_features)
        
        # 将预测结果添加到输入序列中
        input_seq = np.vstack((input_seq, pred))
        predictions.append(pred)
    predictions = np.array(predictions)
    # 反向转换预测结果
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions[:, -1]  # 假设我们只关心最后一个特征
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(data[:, -1], label='Historical Data')  # 假设我们只关心最后一个特征
    plt.plot(range(input_size - 1, input_size + predict_size), np.append(data[input_size-1, -1], predictions), label='Predictions', color='red')
    plt.axvline(input_size - 1, color='grey', linestyle='--')
    plt.legend()
    plt.title("Historical Data and Rolling Predictions")
    plt.show()

    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN-LSTM', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=96, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=4, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='./exp/final_homework/ETTh1.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='OT', help='你需要预测的特征列,这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=7, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=200, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')
    # model
    parser.add_argument('-hidden_size', type=int, default=64, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-laryer_num', type=int, default=2)
    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")
    # option
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=False)
    args = parser.parse_args()
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size
    # 实例化模型
    try:
        model = LSTM(args.input_size, args.output_size, args.pre_len, args.hidden_size , args.laryer_num, args.drop_out).to(device)
        print(f"初始化{args.model}模型成功")
    except:
        print(f"开始初始化{args.model}模型失败")
    # 训练模型
    if args.train:
        print(f"开始{args.model}模型训练")
        train(model, args, scaler, device)
    if args.test:
        print(f"开始{args.model}模型测试")
        test(model, args, test_loader, scaler)
    if args.predict:
        df = pd.read_csv(args.data_path)
        rolling_predict(model, scaler, df, device)
    plt.show()