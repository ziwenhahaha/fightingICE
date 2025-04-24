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
import random
import wandb  # 添加wandb导入
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
from src.dqn import DQNAgent

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

class DQNTrainer:
    """DQN训练器"""
    def __init__(self, agent, env, eval_env=None, config=None):
        """
        初始化DQN训练器
        
        参数:
            agent: DQN智能体
            env: 训练环境
            eval_env: 评估环境，如果为None则使用训练环境
            config: 配置参数字典
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env if eval_env else env
        
        # 使用用户提供的配置或默认配置
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.num_evaluate_episode = self.config['num_evaluate_episode']
        self.buffer_num_before_learning_begin = self.config['buffer_num_before_learning_begin']
        self.ckpt_path = self.config['ckpt_path']
        
        # 是否已初始化
        self.initialized = False
    
    def init_training(self):
        """
        初始化训练（填充经验回放缓冲区）
        """
        state = self.env.reset()
        count = 0
        
        while count < self.buffer_num_before_learning_begin:
            # 随机选择动作
            action = random.randint(0, self.agent.action_dim - 1)
            next_state, reward, done, _ = self.env.step(action)
            
            # 存储经验
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            count += 1
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
        
        self.initialized = True
        print(f"已收集 {count} 个初始样本用于训练")
    
    def train_one_iteration(self, max_steps=1000):
        """
        训练一个迭代
        
        参数:
            max_steps: 最大步数
            
        返回:
            total_reward: 总奖励
            avg_loss: 平均损失
            steps: 迭代步数
            info: 详细训练信息字典
        """
        # 确保已初始化
        if not self.initialized:
            self.init_training()
            
        state = self.env.reset()
        done = False
        total_reward = 0
        losses = []
        steps = 0
        
        # 用于记录这个迭代收集的样本数量
        samples_collected = 0
        # 用于记录是否完成了完整的轨迹(episode)
        episode_completed = False
        
        # 收集每步奖励，用于计算平均步奖励
        step_rewards = []
        
        while not done and steps < max_steps:
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            
            # 累计奖励
            total_reward += reward
            step_rewards.append(reward)
            
            # 存储经验
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            samples_collected += 1
            
            # 如果回放缓冲区中样本足够，则训练
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                loss = self.agent.learn()
                if loss != 0.0:
                    losses.append(loss)
            
            # 更新状态
            state = next_state
            steps += 1
        
        # 如果完成了游戏，则标记为完整轨迹
        if done:
            episode_completed = True
        
        # 计算平均损失
        avg_loss = np.mean(losses) if losses else 0.0
        # 计算平均步奖励
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        # 计算胜负率
        win_rate = 1.0 if total_reward > 0 else 0.0
        lose_rate = 1.0 if total_reward < 0 else 0.0
        draw_rate = 1.0 if total_reward == 0 else 0.0
        
        # 创建要返回的训练信息字典
        info = {
            'avg_loss': avg_loss,
            'avg_step_reward': avg_step_reward,
            'win_rate': win_rate,
            'lose_rate': lose_rate,
            'draw_rate': draw_rate,
            'samples_collected': samples_collected,
            'completed_episodes': 1 if episode_completed else 0,  # 记录是否完成了有效轨迹
            'buffer_size': len(self.agent.replay_buffer),
            'training_count': len(losses)  # 训练次数
        }
        
        return total_reward, avg_loss, steps, info
    
    def evaluate(self):
        """
        评估当前策略
        
        返回:
            avg_reward: 平均奖励
        """
        total_reward = 0
        
        for _ in range(self.num_evaluate_episode):
            state = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.select_action(state, is_training=False)
                state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
    
    def save_model(self, path=None):
        """
        保存模型
        
        参数:
            path: 保存路径，如果为None则使用默认路径
        """
        if path is None:
            path = f"{self.ckpt_path}/policy_net.pt"
            
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.agent.policy_net.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        self.agent.policy_net.load_state_dict(torch.load(path))
        self.agent.update_target_network()
        print(f"模型已从 {path} 加载")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DQN训练FightingICE')
    
    parser.add_argument('--max_iterations', type=int, default=1000, 
                      help='最大训练迭代数')
    parser.add_argument('--max_steps', type=int, default=1000, 
                      help='每个迭代的最大步数')
    parser.add_argument('--eval_interval', type=int, default=10, 
                      help='评估间隔（迭代数）')
    parser.add_argument('--save_interval', type=int, default=50, 
                      help='保存模型间隔（迭代数）')
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
    parser.add_argument('--num_evaluate_episode', type=int, default=3, 
                      help='评估回合数')
    # 添加wandb相关参数
    parser.add_argument('--use_wandb', action='store_true', 
                      help='是否使用wandb记录训练数据')
    parser.add_argument('--wandb_entity', type=str, default='kaiwu-ia', 
                      help='wandb实体名称')
    parser.add_argument('--wandb_project', type=str, default='fightingICE', 
                      help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default='', 
                      help='wandb运行名称')
    
    return parser.parse_args()


def setup_logger(log_path):
    """设置日志记录"""
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return log_file


def train(args):
    """训练主函数"""
    # 需要提前在外面执行 Xvfb :10 -screen 0 1024x768x16 &   执行一次即可
    # 设置虚拟显示器
    # 这行代码是为了在没有显示器的环境中运行GUI程序
    os.environ['DISPLAY'] = ':10'

    # 获取当前脚本的目录
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

    # 设置工作目录为脚本所在的目录
    os.chdir(script_directory)

    # 验证当前工作目录
    current_directory = os.getcwd()
    print(f"当前工作目录已设置为: {current_directory}")

    # 检查CUDA和cuDNN版本
    print("torch版本:   ",torch.__version__)
    print("cuda版本:    ",torch.version.cuda)
    print("cudnn版本:   ",torch.backends.cudnn.version())
    print("cuda能否使用: ",torch.cuda.is_available())
    print("gpu数量:     ",torch.cuda.device_count())
    print("当前设备索引: ",torch.cuda.current_device())
    print("返回gpu名字： ",torch.cuda.get_device_name(0))
    try:
        print("返回gpu名字： ",torch.cuda.get_device_name(1))
    except:
        pass


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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建环境
    env_kwargs = {
        'p2': args.p2,  # 对手AI
        'frameskip': True,  # 启用帧跳过
        'fourframe': True,  # 使用四帧堆叠
        'java_env_path':script_dir # FightingICE的Java环境路径，跟train.py同级
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
    
    try:
        # 记录训练总样本数
        total_samples = 0
        # 记录有效轨迹(episode)总数
        total_episodes = 0
        
        # 训练循环
        for iteration in range(1, args.max_iterations + 1):
            # 训练一个迭代
            start_time = time.time()
            total_reward, avg_loss, steps, train_info = trainer.train_one_iteration(max_steps=args.max_steps)
            
            # 更新总样本数量
            total_samples += train_info['samples_collected']
            # 更新总轨迹数量
            total_episodes += train_info['completed_episodes']
            
            iteration_time = time.time() - start_time
            log_msg = f"迭代: {iteration}/{args.max_iterations} | "
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
            log_msg += f"轨迹数: {train_info['completed_episodes']} | "
            log_msg += f"总轨迹数: {total_episodes} | "
            log_msg += f"总样本数: {total_samples} | "
            log_msg += f"用时: {iteration_time:.2f}s"
            
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")
            
            # 使用 wandb 记录训练数据
            if args.use_wandb:
                # 创建三组数据，分别以 iteration、episode(轨迹) 和 samples 为横坐标
                # 第一组：以 iteration 为横坐标
                wandb_log_data = {
                    'iteration/reward': total_reward,
                    'iteration/loss': avg_loss,
                    'iteration/steps': steps,
                    'iteration/epsilon': agent.epsilon,
                    'iteration/step_reward': train_info['avg_step_reward'],
                    'iteration/win_rate': train_info['win_rate'],
                    'iteration/lose_rate': train_info['lose_rate'],
                    'iteration/draw_rate': train_info['draw_rate'],
                    'iteration/buffer_size': train_info['buffer_size'],
                    'iteration/training_count': train_info['training_count'],
                    'iteration/samples_per_iteration': train_info['samples_collected'],
                    'iteration/episodes_per_iteration': train_info['completed_episodes'],
                    'iteration/time': iteration_time
                }
                
                # 第二组：以 total_episodes(轨迹) 为横坐标
                wandb_log_data.update({
                    'episode/reward': total_reward,
                    'episode/loss': avg_loss,
                    'episode/steps': steps,
                    'episode/epsilon': agent.epsilon,
                    'episode/step_reward': train_info['avg_step_reward'],
                    'episode/win_rate': train_info['win_rate'],
                    'episode/lose_rate': train_info['lose_rate'],
                    'episode/draw_rate': train_info['draw_rate'],
                    'episode/buffer_size': train_info['buffer_size'],
                    'episode/training_count': train_info['training_count'],
                })
                
                # 第三组：以 total_samples 为横坐标
                wandb_log_data.update({
                    'samples/reward': total_reward,
                    'samples/loss': avg_loss,
                    'samples/steps': steps,
                    'samples/epsilon': agent.epsilon,
                    'samples/step_reward': train_info['avg_step_reward'],
                    'samples/win_rate': train_info['win_rate'],
                    'samples/lose_rate': train_info['lose_rate'],
                    'samples/draw_rate': train_info['draw_rate'],
                    'samples/buffer_size': train_info['buffer_size'],
                    'samples/training_count': train_info['training_count'],
                })
                
                # 记录自定义 x 轴数据
                wandb_log_data.update({
                    'iteration': iteration,
                    'total_episodes': total_episodes,
                    'total_samples': total_samples
                })
                
                wandb.log(wandb_log_data)
            
            # 定期评估
            if iteration % args.eval_interval == 0:
                eval_rewards = trainer.evaluate(num_episodes=args.num_evaluate_episode)
                eval_reward = np.mean(eval_rewards)
                eval_msg = f"评估 | 迭代: {iteration}/{args.max_iterations} | 平均奖励: {eval_reward:.2f}"
                print(eval_msg)
                with open(log_file, 'a') as f:
                    f.write(eval_msg + "\n")
                
                # 记录评估结果到 wandb
                if args.use_wandb:
                    wandb.log({
                        'iteration/eval_reward': eval_reward,
                        'episode/eval_reward': eval_reward,
                        'samples/eval_reward': eval_reward,
                        'iteration': iteration,
                        'total_episodes': total_episodes,
                        'total_samples': total_samples
                    })
                
                # 保存最佳模型
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    trainer.save_model(f"{args.ckpt_path}/best_model.pt")
                    best_msg = f"保存最佳模型 | 迭代: {iteration} | 奖励: {best_eval_reward:.2f}"
                    print(best_msg)
                    with open(log_file, 'a') as f:
                        f.write(best_msg + "\n")
            
            # 定期保存模型
            if iteration % args.save_interval == 0:
                trainer.save_model(f"{args.ckpt_path}/policy_net_{iteration}.pt")
        
        # 保存最终模型
        trainer.save_model(f"{args.ckpt_path}/final_model.pt")
        print(f"训练完成，最终模型已保存。")
    
    except KeyboardInterrupt:
        print("训练中断，保存当前模型...")
        trainer.save_model(f"{args.ckpt_path}/interrupt_model.pt")
        print("当前模型已保存。")

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
