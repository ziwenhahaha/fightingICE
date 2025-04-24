import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import sys
import os
# 使用正确的路径导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from src.network import FullyConnectedNet

# 默认超参数设置，与mindspore版本保持一致
DEFAULT_CONFIG = {
    # DQN超参数
    'gamma': 0.9,                      # 折扣因子
    'lr': 0.001,                       # 学习率
    'buffer_capacity': 20000,          # 经验回放缓冲区大小
    'batch_size': 32,                  # 批次大小
    'update_target_iter': 100,         # 目标网络更新频率
    
    # 探索参数
    'epsi_high': 0.9,                  # 初始探索率
    'epsi_low': 0.05,                  # 最终探索率
    'decay': 200,                      # 探索率衰减速度
    
    # 神经网络参数
    'state_space_dim': 144,            # 状态空间维度
    'action_space_dim': 40,            # 动作空间维度
    'hidden_size': 128,                # 隐藏层大小
    
    # 训练参数
    'num_evaluate_episode': 3,         # 评估回合数
    'buffer_num_before_learning_begin': 32,  # 开始学习前的经验收集数量
    'ckpt_path': './models',           # 模型保存路径
}


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        """
        初始化缓冲区
        
        参数:
            capacity: 缓冲区大小
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储经验
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 回合是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样批次
        
        参数:
            batch_size: 批次大小
            
        返回:
            经验样本批次
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # 转换为PyTorch张量
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward))
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done))
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""
    def __init__(self, config=None):
        """
        初始化DQN智能体
        
        参数:
            config: 配置参数字典，如果为None则使用默认配置
        """
        # 使用用户提供的配置或默认配置
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.state_dim = self.config['state_space_dim']
        self.action_dim = self.config['action_space_dim']
        self.hidden_size = self.config['hidden_size']
        self.gamma = self.config['gamma']
        self.lr = self.config['lr']
        self.batch_size = self.config['batch_size']
        self.update_target_iter = self.config['update_target_iter']
        
        # 探索参数
        self.epsilon_high = self.config['epsi_high']
        self.epsilon_low = self.config['epsi_low']
        self.decay = self.config['decay']
        self.epsilon = self.epsilon_high
        
        # 创建策略网络和目标网络
        self.policy_net = FullyConnectedNet(self.state_dim, self.hidden_size, self.action_dim)
        self.target_net = FullyConnectedNet(self.state_dim, self.hidden_size, self.action_dim)
        
        # 复制策略网络参数到目标网络
        self.update_target_network()
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config['buffer_capacity'])
        
        # 设置优化器（使用RMSProp，与mindspore版本保持一致）
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        
        # 训练步数计数器
        self.train_step_counter = 0
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前网络的设备: {self.device} ")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def update_target_network(self):
        """硬更新：直接复制策略网络参数到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, state, is_training=True):
        """
        选择动作（使用epsilon-greedy策略）
        
        参数:
            state: 当前状态
            is_training: 是否处于训练模式
            
        返回:
            选择的动作
        """
        # 对状态进行预处理
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 训练模式下使用epsilon-greedy策略
        if is_training:
            # 根据训练步数计算当前epsilon值
            self.epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * \
                           np.exp(-1.0 * self.train_step_counter / self.decay)
                
            # 随机探索
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
        
        # 贪婪选择最优动作
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def learn(self):
        """
        从经验回放缓冲区中学习
        
        返回:
            损失值
        """
        # 缓冲区样本不足时不进行学习
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 从经验回放缓冲区采样批次
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        print("当前的设备:", self.device)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # 计算当前Q值
        q_values = self.policy_net(state)
        print(f"当前的Q值的形状{q_values.shape},设备{q_values.device}", )
        # print("q_values的设备:", q_values.device)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        # 计算损失
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新训练步数
        self.train_step_counter += 1
        
        # 定期更新目标网络
        if self.train_step_counter % self.update_target_iter == 0:
            self.update_target_network()
            
        return loss.item()


