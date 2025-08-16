import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pandas as pd


warnings.filterwarnings('ignore')
def SSA(series, level):
    series = series - np.mean(series)
    windowLen = level
    seriesLen = len(series)
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = series[i:i + windowLen]
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)
    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)
    res = pd.DataFrame(rec.T, columns=[f'rec_{i + 1}' for i in range(windowLen)])
    return res

def SSA_multidimensional(data, level):
    ssa_results = []
    for dim_index in range(4):
        dim_data = data[:, dim_index]
        ssa_result = SSA(dim_data, level)
        ssa_df = pd.DataFrame(ssa_result)
        ssa_results.append(ssa_df)
    concatenated_ssa = pd.concat(ssa_results, axis=1)
    last_four_columns = data[:, -4:]
    last_four_df = pd.DataFrame(last_four_columns)
    final_result = pd.concat([concatenated_ssa, last_four_df], axis=1)

    final_array = final_result.values
    return final_array

class Solar_PV(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 mask_rate_impute=0, mask_type='except_last',subsequence_num=4,decomposition_method ="ssa"):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.mask_rate_impute = mask_rate_impute
        self.mask_type = mask_type  # 掩码类型参数
        self.root_path = root_path
        self.data_path = data_path
        self.subsequence_num = subsequence_num
        self.decomposition_method = decomposition_method
        self.__read_data__()

    def __preprocessing_SSA(self, data):
        result_ssa = SSA_multidimensional(data,self.subsequence_num)
        return result_ssa

    def __read_data__(self):
        print("mask_rate_impute ",self.mask_rate_impute)
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 18 * 30 * 48 - self.seq_len, 18 * 30 * 48 + 3 * 30 * 48 - self.seq_len]
        border2s = [18 * 30 * 48, 18 * 30 * 48 + 3 * 30 * 48, 18 * 30 * 48 + 6 * 30 * 48]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data].copy()
        elif self.features == 'S':
            df_data = df_raw[[self.target]].copy()
        else:
            raise ValueError(f"Unsupported features type: {self.features}")
        df_data_original = df_data.copy()
        if self.mask_rate_impute > 0 and self.features in ['M', 'MS','S']:
            if self.mask_type == 'except_last':
                cols_to_mask = df_data.columns[:-1]
                mask = np.random.rand(*df_data[cols_to_mask].shape) < self.mask_rate_impute
                df_data_masked = df_data.copy()
                df_data_masked[cols_to_mask] = df_data[cols_to_mask].mask(mask, 0)
            else:
                raise ValueError(f"Unsupported mask_type: {self.mask_type}")
            df_data.loc[border1:border2 - 1] = df_data_masked.loc[border1:border2 - 1]
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_y = data
        else:
            data = df_data.values
            data_y = df_data_original.values
        data = self.__preprocessing_SSA(data)
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc type: {self.timeenc}")
        self.data_x = data[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


