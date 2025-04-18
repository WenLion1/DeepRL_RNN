import torch
from torch import nn
def model_type(model_name: str, input_size: int, hidden_size: int, num_layers: int, batch_first: bool):
    """
    选择模型类型，rnn、lstm还是gru

    :param batch_first: 批次数是否在第一个位置
    :param num_layers: 一次输出中间的层数
    :param hidden_size: 隐藏层维度大小
    :param input_size: 输入维度大小
    :param model_name: rnn、lstm、gru
    :return: 具体的模型
    """

    model = None
    if model_name == "rnn":
        model = nn.RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    elif model_name == "lstm":
        model = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=batch_first, )
    elif model_name == "gru":
        model = nn.GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    else:
        print("模型加载出现错误!")

    return model

class RNN(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=7,
                 hidden_size=32,
                 num_layers=1,
                 batch_first=True,
                 output_size=1,
                 model_name='rnn',
                 is_sigmoid=1,):
        super(RNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.model_name = model_name
        self.is_sigmoid = is_sigmoid

        self.network = model_type(model_name=self.model_name,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)

    def forward(self, difficulty, switch, is_start):
        # difficulty: (batch_size, seq_len, 3)
        # switch: (batch_size, seq_len, 2)
        # is_start: (batch_size, seq_len, 1)
        input = torch.cat((difficulty, switch, is_start), dim=2).to(self.device)
        out, hn = self.network(input)
        pre = self.fc(out)
        if self.is_sigmoid == 1:
            pre = torch.sigmoid(pre)
        return pre, out, hn

if __name__ == "__main__":
    batch_size = 4
    seq_len = 10

    difficulty = torch.zeros(batch_size, seq_len, 3)
    switch = torch.zeros(batch_size, seq_len, 2)

    # 随机生成 one-hot 向量
    for b in range(batch_size):
        for t in range(seq_len):
            difficulty[b, t, torch.randint(0, 3, (1,))] = 1.0
            switch[b, t, torch.randint(0, 2, (1,))] = 1.0

    model = RNN()
    pre, out, hn = model(difficulty, switch)
    print(out.shape)
    print(out)
    print("__________________")
    print(hn.shape)
    print(hn)