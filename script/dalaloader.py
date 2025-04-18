import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def extract_behavioral_data(mat_path,
                            row1='tDev',
                            row2='TF',
                            row3='pr_of_switch',
                            var_name='results'):
    """
    从 .mat 文件中提取 tdev、tf 和 pr_of_switch 三列数据，并添加 is_start 标记。

    参数:
    - mat_path: str，.mat 文件路径
    - var_name: str，.mat 文件中包含行为数据的变量名，默认是 'results'

    返回:
    - tdev_array: np.ndarray，所有被试的 tdev 值
    - tf_array: np.ndarray，所有被试的 tf 值
    - pr_of_switch_array: np.ndarray，所有被试的 pr_of_switch 值
    - is_start_array: np.ndarray，标记每个被试数据起始位置，起始为 1，其余为 0
    """
    data = scipy.io.loadmat(mat_path)

    if var_name not in data:
        raise ValueError(f"变量 '{var_name}' 不在 .mat 文件中，实际变量有：{list(data.keys())}")

    results = data[var_name][0]

    tdev_list = []
    tf_list = []
    pr_of_switch_list = []
    is_start_list = []

    for row in results:
        tdev = row[row1].squeeze()
        tf = row[row2].squeeze()
        pr = row[row3].squeeze()

        n = len(tdev)
        tdev_list.extend(tdev.tolist())
        tf_list.extend(tf.tolist())
        pr_of_switch_list.extend(pr.tolist())

        is_start = [1] + [0] * (n - 1)
        is_start_list.extend(is_start)

    return (
        np.array(tdev_list),
        np.array(tf_list),
        np.array(pr_of_switch_list),
        np.array(is_start_list)
    )


def reshape_own(*arrays):
    return [a.reshape(1, -1) if len(a.shape) == 1 else a for a in arrays]


def split_data_by_ratio(tdev,
                        tf,
                        pr_of_switch,
                        is_start,
                        train_ratio=0.6,
                        valid_ratio=0.2,
                        test_ratio=0.2):
    """
    将数据按比例分为 train，valid 和 test，并统一重塑为 (n_samples, 1)

    :param tdev: np.ndarray
    :param tf: np.ndarray
    :param pr_of_switch: np.ndarray
    :param is_start: np.ndarray
    :param train_ratio: float
    :param valid_ratio: float
    :param test_ratio: float
    :return: 三个元组 (train, valid, test)，每个元组为 (tdev, tf, pr_of_switch, is_start)
    """
    assert len(tdev) == len(tf) == len(pr_of_switch) == len(is_start), "数组长度必须一致"
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    total = len(tdev)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    # 划分
    tdev_train, tdev_valid, tdev_test = tdev[:train_end], tdev[train_end:valid_end], tdev[valid_end:]
    tf_train, tf_valid, tf_test = tf[:train_end], tf[train_end:valid_end], tf[valid_end:]
    pr_train, pr_valid, pr_test = pr_of_switch[:train_end], pr_of_switch[train_end:valid_end], pr_of_switch[valid_end:]
    is_start_train, is_start_valid, is_start_test = is_start[:train_end], is_start[train_end:valid_end], is_start[
                                                                                                         valid_end:]

    # 重塑为 (n_samples, 1)
    tdev_train, tf_train, pr_train, is_start_train = reshape_own(tdev_train, tf_train, pr_train, is_start_train)
    tdev_valid, tf_valid, pr_valid, is_start_valid = reshape_own(tdev_valid, tf_valid, pr_valid, is_start_valid)
    tdev_test, tf_test, pr_test, is_start_test = reshape_own(tdev_test, tf_test, pr_test, is_start_test)

    return (tdev_train, tf_train, pr_train, is_start_train), \
        (tdev_valid, tf_valid, pr_valid, is_start_valid), \
        (tdev_test, tf_test, pr_test, is_start_test)


class RNNInputDataset(Dataset):
    def __init__(self,
                 difficulty,
                 switch,
                 labels,
                 is_start, ):
        """
        difficulty: np.ndarray，形状 (n_samples, seq_len)，元素为 1/2/3
        switch: np.ndarray，形状 (n_samples, seq_len)，元素为 0/1
        is_start: np.ndarray，形状 (n_samples, seq_len)，元素为 0/1，表示每个序列开始位置
        labels: np.ndarray，形状 (n_samples,) 或 (n_samples, 1)，回归目标（如 pr_of_switch）
        """
        self.difficulty = torch.LongTensor(difficulty - 1)  # 将 1/2/3 映射到 0/1/2
        self.switch = torch.LongTensor(switch)
        self.is_start = torch.LongTensor(is_start)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.difficulty)

    def __getitem__(self, idx):
        diff_onehot = torch.nn.functional.one_hot(self.difficulty[idx], num_classes=3)
        switch_onehot = torch.nn.functional.one_hot(self.switch[idx], num_classes=2)
        start_onehot = torch.nn.functional.one_hot(self.is_start[idx], num_classes=2)

        # 可按需拼接成一个输入向量，也可分别返回
        # input_tensor = torch.cat([diff_onehot, switch_onehot, start_onehot], dim=-1).float()

        label = self.labels[idx]
        return diff_onehot, switch_onehot, start_onehot, label


if __name__ == "__main__":
    """
    预测切换概率
    """
    # tdev, tf, pr_of_switch, is_start = extract_behavioral_data('../data/infer_data.mat',
    #                                                            var_name='infer_data')
    """
    预测confidence
    """
    tdev, tf, confidence, is_start = extract_behavioral_data('../data/infer_data.mat',
                                                             row1='tDev',
                                                             row2='TF',
                                                             row3='mu_switch_estimated',
                                                             var_name='infer_data', )
    # (tdev_train, tf_train, pr_train), (tdev_valid, tf_valid, pr_valid), (
    # tdev_test, tf_test, pr_test) = split_data_by_ratio(tdev=tdev,
    #                                                    tf=tf,
    #                                                    pr_of_switch=pr_of_switch,
    #                                                    train_ratio=0.6,
    #                                                    test_ratio=0.2,
    #                                                    valid_ratio=0.2, )
    tdev, tf, pr_of_switch, is_start = reshape_own(tdev, tf, pr_of_switch, is_start)
    print(is_start.shape)
    print(tdev.shape)
