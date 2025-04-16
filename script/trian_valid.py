import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import deque
from script.dalaloader import extract_behavioral_data, RNNInputDataset, split_data_by_ratio
from script.model import RNN

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
                   early_stop_delta=1e-4,):
    """
    训练验证函数

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

        for diff_tensor, switch_tensor, targets in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            diff_tensor, switch_tensor, targets = diff_tensor.float().to(device), switch_tensor.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            pre, out, hn = model(diff_tensor, switch_tensor)  # 拆出 difficulty 和 switch
            targets = targets.transpose(1, 2)
            loss = criterion(pre, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for diff_tensor, switch_tensor, targets in val_loader:
                diff_tensor, switch_tensor, targets = diff_tensor.float().to(device), switch_tensor.float().to(device), targets.float().to(device)
                outputs, _, _ = model(diff_tensor, switch_tensor)
                targets = targets.transpose(1, 2)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping 判断
        if avg_val_loss + early_stop_delta < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0

            # 保存当前最优模型参数
            model_filename = f"best_model.pt"
            best_model_path = os.path.join(save_model_path, model_filename)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs)")
                break


if __name__ == "__main__":
    model = RNN()
    tdev, tf, pr_of_switch = extract_behavioral_data('../data/infer_data.mat', var_name='infer_data')
    (tdev_train, tf_train, pr_train), (tdev_valid, tf_valid, pr_valid), (tdev_test, tf_test, pr_test) = split_data_by_ratio(tdev=tdev,
                                                                                                                            tf=tf,
                                                                                                                            pr_of_switch=pr_of_switch,
                                                                                                                            train_ratio=0.6,
                                                                                                                            test_ratio=0.2,
                                                                                                                            valid_ratio=0.2, )

    dataset_train = RNNInputDataset(tdev_train, tf_train, pr_train)
    dataset_valid = RNNInputDataset(tdev_valid, tf_valid, pr_valid)
    dataset_test = RNNInputDataset(tdev_test, tf_test, pr_test)
    train_val_loop(model=model,
                   dataset_train=dataset_train,
                   dataset_valid=dataset_valid,
                   save_model_path="../model",
                   early_stop_patience=10,)
