import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import deque
from script.dalaloader import extract_behavioral_data, RNNInputDataset, split_data_by_ratio, reshape_own
from script.model import RNN
from script.utils import set_seed, reshape_and_mask_tdev


def train_val_loop(model,
                   dataset_train,
                   dataset_valid,
                   save_model_path,
                   epochs=1000,
                   batch_size=1,
                   lr=1e-3,
                   device='cpu',
                   shuffle=False,
                   early_stop_patience=5,
                   early_stop_delta=1e-4,
                   model_name="best_model.pt", ):
    """
    训练验证函数

    :param model_name:
    :param save_model_path:
    :param model:
    :param dataset_train:
    :param dataset_valid:
    :param epochs:
    :param batch_size:
    :param lr:
    :param device:
    :param shuffle:
    :param early_stop_patience:
    :param early_stop_delta:
    :return:
    """

    os.makedirs(save_model_path, exist_ok=True)

    # 加载器
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset_valid, batch_size=batch_size)

    # 模型、损失、优化器
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for diff_tensor, switch_tensor, is_start_tensor, targets in tqdm(train_loader,
                                                                         desc=f"[Epoch {epoch}] Training"):
            diff_tensor, switch_tensor, is_start_tensor, targets = diff_tensor.float().to(
                device), switch_tensor.float().to(
                device), is_start_tensor.float().to(device), targets.float().to(device)
            # print(diff_tensor.shape)
            # print(switch_tensor.shape)
            # asdasd

            # # 当switch在当前trial为1时，让diff变为[0, 0, 0]
            # diff_tensor[0], switch_tensor[0] = reshape_and_mask_tdev(diff=diff_tensor[0],
            #                                                          switch=switch_tensor[0])

            # print(diff_tensor.shape)
            # print(switch_tensor.shape)
            # print(diff_tensor[0])
            # print(switch_tensor[0])
            # asdasd

            optimizer.zero_grad()
            pre, out, hn = model(diff_tensor, switch_tensor, is_start_tensor)  # 拆出 difficulty 和 switch

            batch_loss = 0.0
            loss_count = 0
            mistake_count = 0  # 记录连续错的次数

            for t in range(pre.size(1)):
                pre_t = pre[:, t, :]
                target_t = targets[:, :, t]

                current_switch = switch_tensor[0][t]  # shape: (2,)
                if torch.all(current_switch == torch.tensor([0.0, 1.0], device=device)):
                    weight = 1.0
                    mistake_count = 0  # 正确，重置
                elif torch.all(current_switch == torch.tensor([1.0, 0.0], device=device)):
                    mistake_count += 1
                    if mistake_count == 1:
                        weight = 3.0
                    elif mistake_count == 2:
                        weight = 6.0
                    else:
                        weight = 10.0
                else:
                    weight = 1.0  # 如果不是严格 [0,1] 或 [1,0]，默认不加权
                    mistake_count = 0  # 也可考虑不重置，取决于你定义是否严格判断

                loss_t = weight * nn.functional.mse_loss(pre_t, target_t)
                batch_loss += loss_t
                loss_count += 1

            batch_loss = batch_loss / loss_count

            # targets = targets.transpose(1, 2)
            #
            # has_inf = torch.isinf(targets[0])
            # has_nan = torch.isnan(targets[0])
            # # 打印结果
            # if has_inf.any():
            #     valid_vals = targets[0][~has_inf & ~has_nan]
            #     max_val = valid_vals.max() if len(valid_vals) > 0 else torch.tensor(0.0)
            #     targets[0][has_inf] = max_val
            #     print("已将 inf 替换为最大值：", max_val.item())
            # if has_nan.any():
            #     print("包含 nan，位置：", torch.nonzero(has_nan, as_tuple=True)[0])
            # #
            # # print(pre[0][0])
            # # print(targets[0][0])
            # loss = criterion(pre, targets)
            # # print(loss)
            # # asdasd
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for diff_tensor, switch_tensor, is_start, targets in val_loader:
                diff_tensor, switch_tensor, is_start, targets = diff_tensor.float().to(
                    device), switch_tensor.float().to(device), is_start.float().to(device), targets.float().to(device)
                outputs, _, _ = model(diff_tensor, switch_tensor, is_start)

                batch_loss = 0.0
                loss_count = 0

                for t in range(outputs.size(1)):
                    pre_t = outputs[:, t, :]
                    target_t = targets[:, :, t]

                    current_switch = switch_tensor[0][t]  # shape: (2,)
                    if torch.all(current_switch == torch.tensor([0.0, 1.0], device=device)):
                        weight = 1.0
                        mistake_count = 0  # 正确，重置
                    elif torch.all(current_switch == torch.tensor([1.0, 0.0], device=device)):
                        mistake_count += 1
                        if mistake_count == 1:
                            weight = 3.0
                        elif mistake_count == 2:
                            weight = 6.0
                        else:
                            weight = 10.0
                    else:
                        weight = 1.0  # 如果不是严格 [0,1] 或 [1,0]，默认不加权
                        mistake_count = 0  # 也可考虑不重置，取决于你定义是否严格判断

                    loss_t = weight * nn.functional.mse_loss(pre_t, target_t)

                    batch_loss += loss_t
                    loss_count += 1

                val_loss += (batch_loss / loss_count).item()

                # targets = targets.transpose(1, 2)
                #
                # has_inf = torch.isinf(targets[0])
                # has_nan = torch.isnan(targets[0])
                # # 打印结果
                # if has_inf.any():
                #     valid_vals = targets[0][~has_inf & ~has_nan]
                #     max_val = valid_vals.max() if len(valid_vals) > 0 else torch.tensor(0.0)
                #     targets[0][has_inf] = max_val
                #     print("已将 inf 替换为最大值：", max_val.item())
                # if has_nan.any():
                #     print("包含 nan，位置：", torch.nonzero(has_nan, as_tuple=True)[0])

                # batch_loss = criterion(outputs, targets)
                # val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping 判断
        if avg_val_loss + early_stop_delta < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0

            # 保存当前最优模型参数
            best_model_path = os.path.join(save_model_path, model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs)")
                break


if __name__ == "__main__":
    set_seed(42)
    model = RNN(is_sigmoid=1)

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

    """
    无valid和test
    """
    tdev_all, tf_all, target_all, is_start_all = reshape_own(tdev, tf, target, is_start)

    # # 加强数据
    # tdev_all = np.tile(tdev_all, (1, 3))
    # tf_all = np.tile(tf_all, (1, 3))
    # target_all = np.tile(target_all, (1, 3))
    # is_start_all = np.tile(is_start_all, (1, 3))

    dataset_all = RNNInputDataset(tdev_all, tf_all, target_all, is_start_all)

    """
    有valid和test
    """
    # (tdev_train, tf_train, pr_train, is_start_train), (tdev_valid, tf_valid, pr_valid, is_start_valid), (
    # tdev_test, tf_test, pr_test, is_start_test) = split_data_by_ratio(tdev=tdev,
    #                                                                   tf=tf,
    #                                                                   pr_of_switch=pr_of_switch,
    #                                                                   is_start=is_start,
    #                                                                   train_ratio=0.6,
    #                                                                   test_ratio=0.2,
    #                                                                   valid_ratio=0.2, )
    # dataset_train = RNNInputDataset(tdev_train, tf_train, pr_train, is_start_train)
    # dataset_valid = RNNInputDataset(tdev_valid, tf_valid, pr_valid, is_start_valid)

    train_val_loop(model=model,
                   dataset_train=dataset_all,
                   dataset_valid=dataset_all,
                   save_model_path="../model",
                   early_stop_patience=10,
                   model_name="best_model_switch_weight_diff0.pt",
                   epochs=10000, )
