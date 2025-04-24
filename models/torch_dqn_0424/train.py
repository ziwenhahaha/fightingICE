# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
DQN training example.
"""

#pylint: disable=C0413
import os
import sys
import time
import argparse
import numpy as np
import torch
from datetime import datetime

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入环境
sys.path.append(os.path.join(current_dir, 'myenv'))
from myenv.fightingice_env import FightingiceEnv

# 直接导入DQN相关组件，不使用相对导入
sys.path.append(os.path.join(current_dir, 'src'))
from src.network import FullyConnectedNet
from src.dqn import DQNAgent, DQNTrainer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DQN训练FightingICE')
    
    parser.add_argument('--max_episodes', type=int, default=1000, 
                      help='最大训练回合数')
    parser.add_argument('--eval_interval', type=int, default=10, 
                      help='评估间隔（回合数）')
    parser.add_argument('--save_interval', type=int, default=50, 
                      help='保存模型间隔（回合数）')
    parser.add_argument('--gamma', type=float, default=0.9, 
                      help='折扣因子')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='批次大小')
    parser.add_argument('--hidden_size', type=int, default=128, 
                      help='隐藏层大小')
    parser.add_argument('--buffer_capacity', type=int, default=20000, 
                      help='经验回放缓冲区大小')
    parser.add_argument('--epsilon_high', type=float, default=0.9, 
                      help='初始探索率')
    parser.add_argument('--epsilon_low', type=float, default=0.05, 
                      help='最终探索率')
    parser.add_argument('--decay', type=float, default=200, 
                      help='探索率衰减系数')
    parser.add_argument('--update_target_iter', type=int, default=100, 
                      help='目标网络更新频率')
    parser.add_argument('--seed', type=int, default=42, 
                      help='随机种子')
    parser.add_argument('--ckpt_path', type=str, default='./models', 
                      help='模型保存路径')
    parser.add_argument('--log_path', type=str, default='./logs', 
                      help='日志保存路径')
    parser.add_argument('--p2', type=str, default='Sandbox', 
                      help='对手AI')
    parser.add_argument('--port', type=int, default=None, 
                      help='游戏端口')
    
    return parser.parse_args()


def setup_logger(log_path):
    """设置日志记录"""
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return log_file


def train(args):
    """训练主函数"""
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建日志文件
    log_file = setup_logger(args.log_path)
    
    # 配置参数
    config = {
        'gamma': args.gamma,
        'lr': args.lr,
        'buffer_capacity': args.buffer_capacity,
        'batch_size': args.batch_size,
        'update_target_iter': args.update_target_iter,
        'epsi_high': args.epsilon_high,
        'epsi_low': args.epsilon_low,
        'decay': args.decay,
        'hidden_size': args.hidden_size,
        'ckpt_path': args.ckpt_path,
        'num_evaluate_episode': 3,
        'buffer_num_before_learning_begin': 32,
    }
    
    # 创建环境
    env_kwargs = {
        'p2': args.p2,  # 对手AI
        'frameskip': True,  # 启用帧跳过
        'fourframe': True,  # 使用四帧堆叠
    }
    if args.port:
        env_kwargs['port'] = args.port
    
    env = FightingiceEnv(**env_kwargs)
    
    # 记录环境信息
    print(f"状态空间维度: {env.observation_space.shape[0]}")
    print(f"动作空间维度: {env.action_space.n}")
    
    # 更新配置中的状态和动作维度
    config['state_space_dim'] = env.observation_space.shape[0]
    config['action_space_dim'] = env.action_space.n
    
    # 创建智能体和训练器
    agent = DQNAgent(config)
    trainer = DQNTrainer(agent, env, config=config)
    
    # 记录训练开始
    with open(log_file, 'w') as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参数配置: {config}\n\n")
    
    # 训练循环
    best_eval_reward = float('-inf')
    
    for episode in range(1, args.max_episodes + 1):
        start_time = time.time()
        
        # 训练一个回合
        total_reward, avg_loss, steps = trainer.train_one_episode()
        
        # 记录训练信息
        episode_time = time.time() - start_time
        log_msg = f"回合: {episode}/{args.max_episodes} | "
        log_msg += f"奖励: {total_reward:.2f} | "
        
        # 添加战斗获胜信息
        if total_reward > 0:
            win_status = "win"
        elif total_reward < 0:
            win_status = "lose"
        else:
            win_status = "平局"
        log_msg += f"结果: {win_status} | "
        
        log_msg += f"损失: {avg_loss:.6f} | "
        log_msg += f"步数: {steps} | "
        log_msg += f"探索率: {agent.epsilon:.4f} | "
        log_msg += f"用时: {episode_time:.2f}s"
        
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        # 定期评估
        if episode % args.eval_interval == 0:
            eval_reward = trainer.evaluate()
            eval_msg = f"评估 | 回合: {episode}/{args.max_episodes} | 平均奖励: {eval_reward:.2f}"
            print(eval_msg)
            with open(log_file, 'a') as f:
                f.write(eval_msg + "\n")
            
            # 保存最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                trainer.save_model(f"{args.ckpt_path}/best_model.pt")
                best_msg = f"保存最佳模型 | 回合: {episode} | 奖励: {best_eval_reward:.2f}"
                print(best_msg)
                with open(log_file, 'a') as f:
                    f.write(best_msg + "\n")
        
        # 定期保存模型
        if episode % args.save_interval == 0:
            trainer.save_model(f"{args.ckpt_path}/policy_net_{episode}.pt")
    
    # 保存最终模型
    trainer.save_model(f"{args.ckpt_path}/final_model.pt")
    print(f"训练完成！最终模型已保存")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
