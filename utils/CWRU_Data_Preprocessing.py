'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    CWRU Bearing Fault Dataset Preprocessing (MAT to Numpy)
******************************************************************************
'''

# Import necessary libraries
import numpy as np
import scipy.io as scio

# To convert labels to one-hot encoding
def convert_to_one_hot(add_label):
    data = np.zeros([add_label.shape[0], 10])
    ind = add_label.squeeze()
    data[np.arange(ind.shape[0]), ind] = 1
    return data

# To convert numpy data from the CWRU raw dataset
def convert_data(data, label_index):
    lis = []
    for i in range(0, data.shape[0], 240):
        inner = data[i: i + 2048].squeeze()
        if inner.shape[0] < 2048:
            break
        lis.append(inner)
    lis = np.array(lis)
    
    # label
    add_label = np.full([lis.shape[0], 1], label_index)
    new_label = convert_to_one_hot(add_label)
    new_lis = np.concatenate([lis, new_label], 1)
    return new_lis

# Load and preprocess CWRU raw data
def load_raw():
    
    print('CWRU data preprocessing started...')
    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_B007_0_122.mat')
    ret_data_0 = convert_data(data['X122_DE_time'], 0)
    new_data = ret_data_0

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_B014_0_189.mat')
    ret_data_1 = convert_data(data['X189_DE_time'], 1)
    new_data = np.concatenate([new_data, ret_data_1], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_B021_0_226.mat')
    ret_data_2 = convert_data(data['X226_DE_time'], 2)
    new_data = np.concatenate([new_data, ret_data_2], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_IR007_0_109.mat')
    ret_data_3 = convert_data(data['X109_DE_time'], 3)
    new_data = np.concatenate([new_data, ret_data_3], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_IR014_0_174.mat')
    ret_data_4 = convert_data(data['X173_DE_time'], 4)
    new_data = np.concatenate([new_data, ret_data_4], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_IR021_0_213.mat')
    ret_data_5 = convert_data(data['X213_DE_time'], 5)
    new_data = np.concatenate([new_data, ret_data_5], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_OR007@6_0_135.mat')
    ret_data_6 = convert_data(data['X135_DE_time'], 6)
    new_data = np.concatenate([new_data, ret_data_6], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_OR014@6_0_201.mat')
    ret_data_7 = convert_data(data['X201_DE_time'], 7)
    new_data = np.concatenate([new_data, ret_data_7], 0)

    data = scio.loadmat('data/CWRU/raw/48k_Drive_End_OR021@6_0_238.mat')
    ret_data_8 = convert_data(data['X238_DE_time'], 8)
    new_data = np.concatenate([new_data, ret_data_8], 0)

    data = scio.loadmat('data/CWRU/raw/normal_0_97.mat')
    ret_data_9 = convert_data(data['X097_DE_time'], 9)
    new_data = np.concatenate([new_data, ret_data_9], 0)
    print(new_data)
    print(new_data.shape)
    
    # Save the processed data as a numpy file
    np.save('data/CWRU/CWRU_Numpy_Data.npy', new_data)
    print('CWRU data preprocessing completed and saved as CWRU_Numpy_Data.npy')

if __name__ == '__main__':
    load_raw()
