import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNet(nn.Module):
    """
    基本的全连接神经网络
    
    参数:
        input_size: 输入层大小
        hidden_size: 隐藏层大小
        output_size: 输出层大小
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)  # 注释掉的第二个隐藏层，与mindspore版本保持一致
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # 注释掉的第二个隐藏层
        x = self.fc3(x)
        return x