import os

import numpy as np
import numpy.lib.recfunctions as rfn
import scipy
import torch
from scipy.io import savemat
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io as io
from script.dalaloader import extract_behavioral_data, split_data_by_ratio, RNNInputDataset, reshape_own
from script.model import RNN
from script.trian_valid import train_val_loop


def test(model,
         dataset_test,
         batch_size=1,
         device='cpu', ):
    """
    测试函数

    :param model:
    :param dataset_test: 测试数据集
    :param batch_size: 批次大小
    :param device: 设备（CPU或GPU）
    :return:
    """

    # 加载器
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    # 模型评估模式
    model.to(device)
    model.eval()

    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():  # 在测试时不需要计算梯度
        for diff_tensor, switch_tensor, targets in tqdm(test_loader, desc="Testing"):
            diff_tensor, switch_tensor, targets = diff_tensor.float().to(device), switch_tensor.float().to(
                device), targets.float().to(device)
            outputs, _, _ = model(diff_tensor, switch_tensor)

            targets = targets.transpose(1, 2)
            loss = criterion(outputs, targets)
            r2 = r2_score(outputs.squeeze(), targets.squeeze())
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss = {avg_test_loss:.4f}")
    print(f"Test R² = {r2:.4f}")

    return avg_test_loss, outputs



def pre_save_in_mat(matlab_path,
                    pre,
                    col_A="tDev",
                    new_col="rnn_pre",
                    is_train_col="is_train",
                    var_name='infer_data'):
    # 加载 MATLAB 数据
    data = scipy.io.loadmat(matlab_path, struct_as_record=False, squeeze_me=True)

    if var_name not in data:
        raise ValueError(f"变量 '{var_name}' 不在 .mat 文件中，实际变量有：{list(data.keys())}")

    results = data[var_name]
    if results.ndim == 0:
        results = [results]  # 只有一个结构体也包装成列表
    elif isinstance(results, np.ndarray):
        results = results.tolist()

    # 统计总 trial 数
    total_trials = sum(getattr(row, col_A).shape[0] for row in results)

    # 创建 is_train 标签（前60%为1，中20%为0，后20%为2）
    n60 = int(total_trials * 0.6)
    n20 = int(total_trials * 0.2)
    is_train_array = np.concatenate([
        np.ones(n60),
        np.zeros(n20),
        np.full(total_trials - n60 - n20, 2)
    ]).astype(int)

    # 顺序分配 rnn_pre 和 is_train 到结构体
    arr_idx = 0
    for i, row in enumerate(results):
        trials = getattr(row, col_A).shape[0]

        # rnn_pre
        chunk = pre[arr_idx: arr_idx + trials]
        if chunk.shape[0] != trials:
            raise ValueError("数组长度不足")
        chunk = chunk.reshape((trials, 1))
        setattr(row, new_col, chunk)

        # is_train
        is_train_chunk = is_train_array[arr_idx: arr_idx + trials].reshape((trials, 1))
        setattr(row, is_train_col, is_train_chunk)

        arr_idx += trials

    if arr_idx != len(pre):
        print(f"⚠️ 警告：还有未分配的数据，共剩余 {len(pre) - arr_idx} 个元素")

    # 保存结构体数组
    savemat("../data/update_data.mat", {var_name: np.array(results)})


if __name__ == "__main__":
    """
    验证模型
    """
    model = RNN()
    model.load_state_dict(torch.load("../model/best_model.pt"))
    tdev, tf, pr_of_switch = extract_behavioral_data('../data/infer_data.mat', var_name='infer_data')
    (tdev_train, tf_train, pr_train), (tdev_valid, tf_valid, pr_valid), (
        tdev_test, tf_test, pr_test) = split_data_by_ratio(tdev=tdev,
                                                           tf=tf,
                                                           pr_of_switch=pr_of_switch,
                                                           train_ratio=0.6,
                                                           test_ratio=0.2,
                                                           valid_ratio=0.2, )
    tdev_reshape, tf_reshape, pr_reshape = reshape_own(tdev, tf, pr_of_switch)

    dataset_test = RNNInputDataset(tdev_test, tf_test, pr_test)
    dataset_all = RNNInputDataset(tdev_reshape, tf_reshape, pr_reshape)
    # avg_test_loss, outputs = test(model=model,
    #                               dataset_test=dataset_test, )
    avg_all_loss, outputs = test(model=model,
                                 dataset_test=dataset_all, )
    outputs = outputs.squeeze()

    """
    将模型预测结果保存到mat中
    """
    pre_save_in_mat("../data/infer_data.mat",
                    outputs, )
