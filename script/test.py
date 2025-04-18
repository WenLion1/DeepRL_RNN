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
         device='cpu'):
    """
    测试函数

    :param model: 训练好的模型
    :param dataset_test: 测试数据集
    :param batch_size: 批次大小
    :param device: 'cpu' 或 'cuda'
    :return: 平均测试损失、模型输出、隐藏层状态列表
    """

    test_loader = DataLoader(dataset_test, batch_size=batch_size)
    model.to(device)
    model.eval()

    test_loss = 0.0
    criterion = nn.MSELoss()

    all_outputs = []
    all_targets = []
    all_hidden_states = []

    with torch.no_grad():
        for diff_tensor, switch_tensor, is_start, targets in tqdm(test_loader, desc="Testing"):
            diff_tensor = diff_tensor.float().to(device)
            switch_tensor = switch_tensor.float().to(device)
            is_start = is_start.float().to(device)
            targets = targets.float().to(device)

            outputs, hidden_states, _ = model(diff_tensor, switch_tensor, is_start)

            # 保存每个 batch 的输出和隐藏状态
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_hidden_states.append(hidden_states.cpu())

            targets = targets.transpose(1, 2)

            has_inf = torch.isinf(targets[0])
            has_nan = torch.isnan(targets[0])
            # 打印结果
            if has_inf.any():
                valid_vals = targets[0][~has_inf & ~has_nan]
                max_val = valid_vals.max() if len(valid_vals) > 0 else torch.tensor(0.0)
                targets[0][has_inf] = max_val
                print("已将 inf 替换为最大值：", max_val.item())
            if has_nan.any():
                print("包含 nan，位置：", torch.nonzero(has_nan, as_tuple=True)[0])

            loss = criterion(outputs, targets)
            r2 = r2_score(outputs.squeeze().cpu(), targets.squeeze().cpu())
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss = {avg_test_loss:.4f}")
    print(f"Test R² = {r2:.4f}")

    # 拼接所有 batch 的输出和隐藏状态
    all_outputs = torch.cat(all_outputs, dim=0)
    all_hidden_states = torch.cat(all_hidden_states, dim=0)

    return avg_test_loss, all_outputs, all_hidden_states


def pre_save_in_mat(matlab_path,
                    pre,
                    col_A="tDev",
                    new_col="rnn_pre",
                    is_train_col="is_train",
                    hidden_col="hidden_state",
                    hidden_states=None,
                    var_name='infer_data',
                    save_path="../data/update_data.mat",):
    """
    保存预测结果和隐藏状态到 .mat 文件中

    :param save_path:
    :param matlab_path: 原始 .mat 文件路径
    :param pre: 模型预测结果（1D 或 2D数组）
    :param col_A: 原始数据中用于计算 trial 数的列名
    :param new_col: 要保存的预测列名
    :param is_train_col: 要保存的训练标签列名
    :param hidden_col: 要保存的隐藏状态列名
    :param hidden_states: 全部隐藏状态，shape=(total_trials, hidden_dim)
    :param var_name: .mat 文件中的变量名
    """
    import scipy.io
    from scipy.io import savemat

    # 加载数据
    data = scipy.io.loadmat(matlab_path, struct_as_record=False, squeeze_me=True)

    if var_name not in data:
        raise ValueError(f"变量 '{var_name}' 不在 .mat 文件中，实际变量有：{list(data.keys())}")

    results = data[var_name]
    if results.ndim == 0:
        results = [results]
    elif isinstance(results, np.ndarray):
        results = results.tolist()

    total_trials = sum(getattr(row, col_A).shape[0] for row in results)

    # 创建 is_train 标签
    n60 = int(total_trials * 0.6)
    n20 = int(total_trials * 0.2)
    is_train_array = np.concatenate([
        np.ones(n60),
        np.zeros(n20),
        np.full(total_trials - n60 - n20, 2)
    ]).astype(int)

    # 遍历结构体
    arr_idx = 0
    for row in results:
        trials = getattr(row, col_A).shape[0]

        # 处理预测结果
        pred_chunk = pre[arr_idx: arr_idx + trials]
        pred_chunk = pred_chunk.reshape((trials, -1))  # 保证二维
        setattr(row, new_col, pred_chunk)

        # is_train
        is_train_chunk = is_train_array[arr_idx: arr_idx + trials].reshape((trials, 1))
        setattr(row, is_train_col, is_train_chunk)

        # hidden_state
        if hidden_states is not None:
            hidden_chunk = hidden_states[arr_idx: arr_idx + trials]
            setattr(row, hidden_col, hidden_chunk)

        arr_idx += trials

    if arr_idx != len(pre):
        print(f"警告：还有未分配的数据，共剩余 {len(pre) - arr_idx} 个元素")

    # 保存到新 .mat 文件
    savemat(save_path, {var_name: np.array(results)})


if __name__ == "__main__":
    """
    验证模型
    """
    model = RNN(is_sigmoid=1)
    model.load_state_dict(torch.load("../model/best_model_switch_weight_10.pt"))

    """
    预测切换概率
    """
    tdev, tf, target, is_start = extract_behavioral_data('../data/infer_data.mat',
                                                               var_name='infer_data')
    """
    预测confidence
    """
    # tdev, tf, target, is_start = extract_behavioral_data('../data/infer_data.mat',
    #                                                      row1='tDev',
    #                                                      row2='TF',
    #                                                      row3='mu_switch_estimated',
    #                                                      var_name='infer_data', )

    # 无valid和test
    tdev_reshape_all, tf_reshape_all, target_reshape_all, is_start_all = reshape_own(tdev, tf, target, is_start)
    dataset_all = RNNInputDataset(tdev_reshape_all, tf_reshape_all, target_reshape_all, is_start_all)

    # 有valid和test
    # (tdev_train, tf_train, pr_train, is_start_train), (tdev_valid, tf_valid, pr_valid, is_start_valid), (
    #     tdev_test, tf_test, pr_test, is_start_test) = split_data_by_ratio(tdev=tdev,
    #                                                                       tf=tf,
    #                                                                       pr_of_switch=pr_of_switch,
    #                                                                       is_start=is_start,
    #                                                                       train_ratio=0.6,
    #                                                                       test_ratio=0.2,
    #                                                                       valid_ratio=0.2, )
    # dataset_test = RNNInputDataset(tdev_test, tf_test, pr_test, is_start_test)

    avg_all_loss, outputs, all_hidden_state = test(model=model,
                                                   dataset_test=dataset_all, )
    all_hidden_state = all_hidden_state.squeeze()
    outputs = outputs.squeeze()

    """
    将模型预测结果保存到mat中
    """
    pre_save_in_mat("../data/infer_data.mat",
                    outputs,
                    hidden_states=all_hidden_state,
                    save_path="../data/update_data_switch_weight_10.mat")
