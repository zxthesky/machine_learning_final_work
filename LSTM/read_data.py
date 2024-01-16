import numpy as np

def read_data(path):
    data = np.load(path)
    return data

if __name__ == '__main__':
    temp_read_x = read_data('./exp/final_homework/ETTh1/X_train.npy')
    temp_read_y_96 = read_data('./exp/final_homework/ETTh1/y_train_96.npy')
    temp_read_y_336 = read_data('./exp/final_homework/ETTh1/y_train_336.npy')
    print(temp_read_x.shape)
    import pdb; pdb.set_trace()