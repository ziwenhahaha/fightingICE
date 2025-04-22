"""
并行训练脚本 - 使用带日志功能的并行环境
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.vec_env import make_vec_env_with_logging

# 定义一个简单的DQN模型作为演示
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train():
    parser = argparse.ArgumentParser(description='并行训练示例')
    parser.add_argument('--num_envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--episodes', type=int, default=100, help='训练回合数')
    parser.add_argument('--steps_per_episode', type=int, default=200, help='每个回合的最大步数')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志输出目录')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon_start', type=float, default=0.9, help='起始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='最终探索率')
    parser.add_argument('--epsilon_decay', type=int, default=200, help='探索率衰减')
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建带日志功能的并行环境
    print(f"创建 {args.num_envs} 个并行环境...")
    env = make_vec_env_with_logging(
        'FightingiceEnv', 
        n_envs=args.num_envs, 
        log_dir=args.log_dir,
        p2="MctsAi"  # 指定对手
    )
    
    # 获取观察和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建策略网络
    policy_net = DQN(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 训练主循环
    start_time = time.time()
    total_steps = 0
    
    try:
        print("开始训练...")
        for episode in range(args.episodes):
            # 显示当前回合和探索率
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                      np.exp(-1. * total_steps / args.epsilon_decay)
            
            print(f"\n回合 {episode+1}/{args.episodes} - 探索率: {epsilon:.4f}")
            
            # 重置环境
            states = env.reset()
            episode_rewards = np.zeros(args.num_envs)
            
            for step in range(args.steps_per_episode):
                # 选择动作 - epsilon-greedy策略
                if np.random.random() < epsilon:
                    # 随机探索
                    actions = [env.action_space.sample() for _ in range(args.num_envs)]
                else:
                    # 使用策略网络
                    with torch.no_grad():
                        q_values = policy_net(torch.FloatTensor(states))
                        actions = q_values.max(1)[1].numpy()
                
                # 执行动作
                next_states, rewards, dones, infos = env.step(actions)
                
                # 累计奖励
                episode_rewards += rewards
                
                # 简单的Q学习更新
                # 通常我们会使用经验回放，这里简化为直接更新
                states_tensor = torch.FloatTensor(states)
                next_states_tensor = torch.FloatTensor(next_states)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
                
                # 计算当前Q值
                current_q = policy_net(states_tensor).gather(1, torch.LongTensor(actions).unsqueeze(1))
                
                # 计算下一状态的Q值
                next_q = policy_net(next_states_tensor).max(1)[0].unsqueeze(1)
                
                # 计算目标Q值
                target_q = rewards_tensor + (1 - dones_tensor) * args.gamma * next_q
                
                # 计算损失
                loss = criterion(current_q, target_q)
                
                # 优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                states = next_states
                total_steps += 1
                
                # 如果所有环境都结束，则提前结束本回合
                if all(dones):
                    break
                
                # 每20步输出一次训练进度
                if step % 20 == 0 or step == args.steps_per_episode - 1:
                    avg_reward = np.mean(episode_rewards)
                    print(f"回合 {episode+1} - 步数 {step}/{args.steps_per_episode} - "
                          f"平均奖励: {avg_reward:.2f} - 损失: {loss.item():.4f}")
            
            # 每回合结束时保存模型
            if (episode + 1) % 10 == 0:
                model_path = os.path.join(args.save_dir, f"dqn_model_ep{episode+1}.pt")
                torch.save(policy_net.state_dict(), model_path)
                print(f"模型已保存到 {model_path}")
        
        # 训练结束，保存最终模型
        final_model_path = os.path.join(args.save_dir, "dqn_model_final.pt")
        torch.save(policy_net.state_dict(), final_model_path)
        print(f"最终模型已保存到 {final_model_path}")
        
        # 训练结束统计
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
        print(f"总步数: {total_steps}")
        print(f"平均每秒步数: {total_steps/training_time:.2f}")
    
    finally:
        # 确保环境正确关闭
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    train()