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
DQN eval example.
"""
import os
import sys
import argparse
import numpy as np
import torch

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
from src.dqn import DQNAgent, DEFAULT_CONFIG


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估FightingICE DQN模型')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型路径')
    parser.add_argument('--episodes', type=int, default=10,
                      help='评估回合数')
    parser.add_argument('--p2', type=str, default='Sandbox',
                      help='对手AI')
    parser.add_argument('--port', type=int, default=None,
                      help='游戏端口')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--render', action='store_true',
                      help='是否渲染游戏画面')
    
    return parser.parse_args()


def evaluate(args):
    """评估主函数"""
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境
    env_kwargs = {
        'p2': args.p2,
        'frameskip': True,
        'fourframe': True,
    }
    if args.port:
        env_kwargs['port'] = args.port
    
    env = FightingiceEnv(**env_kwargs)
    
    # 记录环境信息
    print(f"状态空间维度: {env.observation_space.shape[0]}")
    print(f"动作空间维度: {env.action_space.n}")
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    config['state_space_dim'] = env.observation_space.shape[0]
    config['action_space_dim'] = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(config)
    
    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        return
    
    agent.policy_net.load_state_dict(torch.load(args.model_path))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())  # 同步目标网络
    print(f"成功加载模型: {args.model_path}")
    
    # 评估
    total_reward = 0
    win_count = 0
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, is_training=False)  # 使用贪婪策略
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        # 检查是否获胜
        if episode_reward > 0:
            win_status = "胜利"
            win_count += 1
        elif episode_reward == 0:
            win_status = "平局"
        else:
            win_status = "失败"
            
        print(f"回合 {episode}/{args.episodes} | 奖励: {episode_reward:.2f} | 结果: {win_status}")
        total_reward += episode_reward
    
    # 计算胜率
    win_rate = win_count / args.episodes * 100
    avg_reward = total_reward / args.episodes
    
    print(f"\n评估完成 | 平均奖励: {avg_reward:.2f} | 胜率: {win_rate:.2f}%")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
