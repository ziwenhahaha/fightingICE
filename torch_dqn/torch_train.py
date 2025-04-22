"""
DQN训练脚本 - PyTorch版本
使用错误处理机制来优雅处理环境故障
"""
import os
import sys
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

# 导入自定义模块
from src.vec_env import make_vec_env_with_logging

# 设置随机种子以便结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 简单的DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, device, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 learning_rate=0.001, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # 网络和优化器
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 训练步骤计数器
        self.train_steps = 0
    
    def select_action(self, state_batch, valid_indices=None):
        """为多个状态选择动作，支持只对有效环境选择动作"""
        actions = []
        
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(state_batch).to(self.device)
        
        # 使用ε-贪婪策略选择动作
        if np.random.random() < self.epsilon:
            # 随机选择动作
            actions = np.random.randint(0, self.action_size, size=len(state_batch))
        else:
            # 使用网络选择动作
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                actions = q_values.max(1)[1].cpu().numpy()
        
        return actions
    
    def save_experience(self, states, actions, rewards, next_states, dones, valid_indices):
        """保存有效环境的经验到回放缓冲区"""
        for i in valid_indices:
            self.memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def train(self):
        """训练DQN网络"""
        # 如果缓冲区没有足够的样本，直接返回
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从缓冲区采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        
        # 计算当前Q值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失并更新网络
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新训练步骤计数器
        self.train_steps += 1
        
        # 更新目标网络（每100步更新一次）
        if self.train_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_steps = checkpoint['train_steps']
            print(f"模型已从 {path} 加载")
            return True
        else:
            print(f"找不到模型文件: {path}")
            return False


def train():
    """训练入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PyTorch DQN训练脚本')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--steps_per_episode', type=int, default=5000, help='每回合最大步数')
    parser.add_argument('--num_envs', type=int, default=20, help='并行环境数量')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=10000, help='回放缓冲区大小')
    parser.add_argument('--target_update', type=int, default=10, help='目标网络更新频率(回合)')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='模型保存频率(回合)')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='探索率衰减')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--timeout', type=int, default=30, help='环境响应超时时间(秒)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"训练设备: {args.device}")
    
    # 确保路径存在
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置日志文件
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'train_log_{current_time}.txt')
    
    # 创建环境
    print(f"创建 {args.num_envs} 个并行环境...")
    env = make_vec_env_with_logging(
        'FightingiceEnv',
        n_envs=args.num_envs,
        log_dir=args.log_dir,
        timeout=args.timeout
    )
    
    try:
        # 获取环境信息
        obs_shape = env.observation_space.shape[0]  # 状态空间维度
        n_actions = env.action_space.n  # 动作空间大小
        print(f"状态空间大小: {obs_shape}")
        print(f"动作空间大小: {n_actions}")
        
        # 创建DQN代理
        agent = DQNAgent(
            state_size=obs_shape,
            action_size=n_actions,
            device=args.device,
            gamma=args.gamma,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            epsilon_decay=args.epsilon_decay
        )
        
        # 恢复训练
        start_episode = 0
        if args.resume:
            if agent.load(args.resume):
                print(f"从检查点恢复训练: {args.resume}")
                # 尝试从文件名解析回合数
                try:
                    filename = os.path.basename(args.resume)
                    if 'episode_' in filename:
                        start_episode = int(filename.split('episode_')[1].split('.')[0])
                        print(f"从回合 {start_episode} 继续训练")
                except:
                    print("无法从文件名解析回合数，从头开始训练")
        
        # 重置环境
        print("重置环境...")
        states = env.reset()
        
        # 检查有效环境数量
        valid_count = env.get_valid_env_count()
        valid_indices = env.get_valid_env_indices()
        print(f"重置后的有效环境数量: {valid_count}/{args.num_envs}")
        
        if valid_count == 0:
            print("错误: 所有环境都无效，无法继续训练")
            return
        
        # 记录训练历史
        episode_rewards = []
        losses = []
        
        # 开始训练
        print(f"开始训练，目标回合数: {args.episodes}")
        start_time = time.time()
        
        for episode in range(start_episode, start_episode + args.episodes):
            episode_reward = np.zeros(args.num_envs)
            episode_start_time = time.time()
            
            # 重置环境
            try:
                states = env.reset()
                # 更新有效环境信息
                valid_count = env.get_valid_env_count()
                valid_indices = env.get_valid_env_indices()
                print(f"回合 {episode+1} 重置后的有效环境数量: {valid_count}/{args.num_envs}")
                
                if valid_count == 0:
                    print("错误: 所有环境都无效，跳过此回合")
                    continue
                
            except Exception as e:
                print(f"重置环境时发生错误: {e}")
                continue
            
            # 显示环境状态摘要
            status_summary = env.get_env_status_summary()
            print(f"环境状态: 总数={status_summary['total']}, "
                  f"有效={status_summary['valid']} ({status_summary['valid_percent']})")
            
            # 单个回合的训练
            total_loss = 0
            n_losses = 0
            
            for step in range(args.steps_per_episode):
                # 选择动作
                actions = agent.select_action(states)
                
                # 执行动作
                try:
                    next_states, rewards, dones, infos = env.step(actions)
                    
                    # 更新有效环境信息
                    valid_indices = env.get_valid_env_indices()
                    valid_count = len(valid_indices)
                    
                    if valid_count == 0:
                        print(f"警告: 所有环境都已无效，结束本回合")
                        break
                    
                except Exception as e:
                    print(f"执行动作时发生错误: {e}")
                    break
                
                # 保存经验到回放缓冲区
                agent.save_experience(states, actions, rewards, next_states, dones, valid_indices)
                
                # 只累加有效环境的奖励
                for i in valid_indices:
                    episode_reward[i] += rewards[i]
                
                # 训练网络
                if len(agent.memory) > agent.batch_size:
                    loss = agent.train()
                    total_loss += loss
                    n_losses += 1
                
                # 更新状态
                states = next_states
                
                # 每100步显示一次进度
                if step % 100 == 0:
                    # 只计算有效环境的平均奖励
                    valid_rewards = [episode_reward[i] for i in valid_indices]
                    mean_reward = np.mean(valid_rewards) if valid_rewards else 0
                    
                    mean_loss = total_loss / n_losses if n_losses > 0 else 0
                    print(f"回合 {episode+1}, 步骤 {step}/{args.steps_per_episode}, "
                          f"有效环境: {valid_count}/{args.num_envs}, "
                          f"平均奖励: {mean_reward:.2f}, 损失: {mean_loss:.4f}, "
                          f"探索率: {agent.epsilon:.4f}")
                
                # 如果所有环境都完成了，提前结束此回合
                if all(dones[i] for i in valid_indices):
                    print(f"所有有效环境都已完成，提前结束回合")
                    break
            
            # 回合结束，记录统计信息
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            
            # 只考虑有效环境的奖励
            valid_rewards = [episode_reward[i] for i in valid_indices]
            if valid_rewards:
                mean_reward = np.mean(valid_rewards)
                episode_rewards.append(mean_reward)
                
                mean_loss = total_loss / n_losses if n_losses > 0 else 0
                losses.append(mean_loss)
                
                print(f"回合 {episode+1} 完成: 平均奖励 = {mean_reward:.2f}, "
                      f"平均损失 = {mean_loss:.4f}, 用时 = {episode_duration:.2f}秒, "
                      f"有效环境 = {valid_count}/{args.num_envs}")
                
                # 写入日志文件
                with open(log_file, 'a') as f:
                    f.write(f"回合 {episode+1}, 奖励: {mean_reward:.2f}, 损失: {mean_loss:.4f}, "
                            f"有效环境: {valid_count}/{args.num_envs}, 探索率: {agent.epsilon:.4f}\n")
            else:
                print(f"回合 {episode+1} 无有效数据，跳过")
            
            # 保存模型
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'dqn_episode_{episode+1}.pt')
                agent.save(checkpoint_path)
        
        # 训练结束
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n训练完成! 总用时: {total_time:.2f}秒")
        
        # 保存最终模型
        final_checkpoint_path = os.path.join(args.checkpoint_dir, 'dqn_final.pt')
        agent.save(final_checkpoint_path)
        
        # 显示训练统计信息
        if episode_rewards:
            print(f"平均回合奖励: {np.mean(episode_rewards):.2f}")
            print(f"平均回合损失: {np.mean(losses):.4f}")
        
        # 显示环境状态摘要
        status_summary = env.get_env_status_summary()
        print("\n环境状态汇总:")
        print(f"总环境数: {status_summary['total']}")
        print(f"有效环境数: {status_summary['valid']} ({status_summary['valid_percent']})")
    
    finally:
        # 确保环境正确关闭
        env.close()
        print("所有环境已关闭")


if __name__ == "__main__":
    train()