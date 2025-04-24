"""
DQN并行训练实现，使用vec_env中的并行环境来加速训练
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from datetime import datetime
import wandb  # 添加 wandb 导入


# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入环境
sys.path.append(os.path.join(current_dir, 'myenv'))
from myenv.fightingice_env import FightingiceEnv

# 导入DQN相关组件
sys.path.append(os.path.join(current_dir, 'src'))
from src.network import FullyConnectedNet
from src.dqn import DQNAgent
from src.vec_env import make_vec_env_with_logging


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DQN并行训练FightingICE')
    
    parser.add_argument('--num_envs', type=int, default=30, 
                      help='并行环境数量')
    parser.add_argument('--max_iterations', type=int, default=1000, 
                      help='最大迭代次数')
    parser.add_argument('--max_steps', type=int, default=1000, 
                      help='每回合最大步数')
    parser.add_argument('--eval_interval', type=int, default=10, 
                      help='评估间隔（迭代次数）')
    parser.add_argument('--save_interval', type=int, default=50, 
                      help='保存模型间隔（迭代次数）')
    parser.add_argument('--gamma', type=float, default=0.95, 
                      help='折扣因子')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, 
                      help='批次大小')
    parser.add_argument('--hidden_size', type=int, default=128, 
                      help='隐藏层大小')
    parser.add_argument('--buffer_capacity', type=int, default=5000, 
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
    parser.add_argument('--p2', type=str, default='MctsAi', 
                      help='对手AI')
    parser.add_argument('--batch_reset_size', type=int, default=10,
                      help='每批重启的环境数量，0表示全部一次性重启')
    parser.add_argument('--train_sample_ratio', type=float, default=10.0,
                      help='训练采样比，每个样本平均用来训练的次数，影响更新频率')
    # 添加wandb相关参数
    parser.add_argument('--use_wandb', action='store_true',
                      help='是否使用wandb记录训练数据')
    parser.add_argument('--wandb_entity', type=str, default='kaiwu-ia',
                      help='wandb实体名称')
    parser.add_argument('--wandb_project', type=str, default='fightingICE',
                      help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default='train_sample_ratio-10.0',
                      help='wandb运行名称')
    
    return parser.parse_args()


def setup_logger(log_path):
    """设置日志记录"""
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return log_file


def train(args):
    """训练主函数"""
    # 设置虚拟显示器
    os.environ['DISPLAY'] = ':10'

    # 获取当前脚本的目录
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

    # 设置工作目录为脚本所在的目录
    os.chdir(script_directory)

    # 验证当前工作目录
    current_directory = os.getcwd()
    print(f"当前工作目录已设置为: {current_directory}")

    # 如果使用wandb，初始化wandb
    if args.use_wandb:
        wandb_run_name = args.wandb_name if args.wandb_name else f"torch-dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "max_steps": args.max_steps,
                "gamma": args.gamma,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "buffer_capacity": args.buffer_capacity,
                "epsilon_high": args.epsilon_high,
                "epsilon_low": args.epsilon_low,
                "decay": args.decay,
                "update_target_iter": args.update_target_iter,
                "p2": args.p2,
                "train_sample_ratio": args.train_sample_ratio
            }
        )
        print(f"WandB已初始化，项目名: {args.wandb_project}, 运行名: {wandb_run_name}")

    # 检查CUDA和cuDNN版本
    print("torch版本:   ", torch.__version__)
    print("cuda版本:    ", torch.version.cuda)
    print("cudnn版本:   ", torch.backends.cudnn.version())
    print("cuda能否使用: ", torch.cuda.is_available())
    print("gpu数量:     ", torch.cuda.device_count())
    print("当前设备索引: ", torch.cuda.current_device())
    print("返回gpu名字： ", torch.cuda.get_device_name(0))
    try:
        print("返回gpu名字： ", torch.cuda.get_device_name(1))
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
        'buffer_num_before_learning_begin': args.batch_size,  # 至少收集一个batch
        'p2': args.p2,  # 对手AI
        'batch_reset_size': args.batch_reset_size,  # 添加批量重启环境大小参数
        'train_sample_ratio': args.train_sample_ratio,  # 训练采样比
    }
    
    with open(log_file, 'w') as f:
        f.write(f"并行训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参数配置: {config}\n")
        f.write(f"并行环境数量: {args.num_envs}\n\n")
        if args.batch_reset_size > 0:
            f.write(f"批量重启大小: {args.batch_reset_size}\n")
        else:
            f.write(f"使用全部环境同时重启\n")
    
    # 创建并行环境（带日志功能）
    env_log_dir = os.path.join(args.log_path, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)
    
    print(f"创建并行环境，数量: {args.num_envs}")
    vec_env = make_vec_env_with_logging(
        'FightingiceEnv',
        n_envs=args.num_envs,
        log_dir=env_log_dir,
        p2=args.p2,
        frameskip=True,
        fourframe=True,
        batch_reset_size=args.batch_reset_size  # 传递批量重启参数
    )
    
    # 获取状态和动作空间维度
    state_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.n
    
    print(f"状态空间维度: {state_dim}")
    print(f"动作空间维度: {action_dim}")
    
    # 更新配置中的状态和动作维度
    config['state_space_dim'] = state_dim
    config['action_space_dim'] = action_dim
    
    # 创建智能体
    agent = DQNAgent(config)
    
    # 创建并行训练器
    trainer = ParallelDQNTrainer(agent, vec_env, config=config, log_file=log_file)
    
    # 训练循环
    best_eval_reward = float('-inf')
    
    try:
        # 记录训练总样本数
        total_samples = 0
        # 记录有效轨迹(episode)总数量
        total_episodes = 0
        
        for iteration in range(1, args.max_iterations + 1):
            start_time = time.time()
            
            # 训练一个迭代
            total_reward, avg_loss, steps, train_info = trainer.train_one_iteration(max_steps=args.max_steps)
            
            # 更新总样本数
            total_samples += train_info['samples_collected']
            # 更新总轨迹数(完成的有效轨迹)
            total_episodes += train_info['completed_episodes']
            
            # 记录训练信息
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
                eval_reward = trainer.evaluate()
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
                    trainer.save_model(f"{args.ckpt_path}/best_model_parallel.pt")
                    best_msg = f"保存最佳模型 | 迭代: {iteration} | 奖励: {best_eval_reward:.2f}"
                    print(best_msg)
                    with open(log_file, 'a') as f:
                        f.write(best_msg + "\n")
            
            # 定期保存模型
            if iteration % args.save_interval == 0:
                trainer.save_model(f"{args.ckpt_path}/policy_net_parallel_{iteration}.pt")
        
        # 保存最终模型
        trainer.save_model(f"{args.ckpt_path}/final_model_parallel.pt")
        print(f"训练完成，最终模型已保存。")
    
    except KeyboardInterrupt:
        print("训练中断，保存当前模型...")
        trainer.save_model(f"{args.ckpt_path}/interrupt_model_parallel.pt")
        print("当前模型已保存。")
    
    finally:
        # 关闭并行环境
        vec_env.close()
        print("并行环境已关闭。")
        
        # 如果使用了 wandb，关闭 wandb
        if args.use_wandb:
            wandb.finish()
            print("WandB已关闭。")


class ParallelDQNTrainer:
    """并行DQN训练器，使用多个环境实例并行收集数据"""
    def __init__(self, agent, vec_env, config=None, log_file=None):
        """
        初始化并行DQN训练器
        
        参数:
            agent: DQN智能体
            vec_env: 并行环境
            config: 配置参数字典
            log_file: 日志文件路径
        """
        self.agent = agent
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.log_file = log_file
        
        # 使用默认配置
        self.config = {
            'gamma': 0.9,
            'lr': 0.001,
            'batch_size': 32,
            'update_target_iter': 100,
            'buffer_capacity': 20000,
            'num_evaluate_episode': 3,
            'buffer_num_before_learning_begin': 32,
            'ckpt_path': './models',
            'train_sample_ratio': 1.0,  # 默认训练采样比
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 参数
        self.num_evaluate_episode = self.config['num_evaluate_episode']
        self.buffer_num_before_learning_begin = self.config['buffer_num_before_learning_begin']
        self.ckpt_path = self.config['ckpt_path']
        self.train_sample_ratio = self.config['train_sample_ratio']  # 训练采样比
        
        # 是否已初始化
        self.initialized = False
        
        # 创建评估用的单环境
        self.eval_env = FightingiceEnv(p2=config.get('p2', 'MctsAi'), frameskip=True, fourframe=True)
        
        # 记录日志
        self.log_message(f"并行训练器已创建，使用 {self.num_envs} 个并行环境")
        self.log_message(f"配置参数: {self.config}")
        self.log_message(f"训练采样比: {self.train_sample_ratio}, 每个样本平均被训练 {self.train_sample_ratio} 次")
    
    def log_message(self, message):
        """记录日志信息"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")
    
    def init_training(self):
        """
        初始化训练（填充经验回放缓冲区）
        使用并行环境快速收集样本
        """
        # 重置所有环境
        states = self.vec_env.reset()
        count = 0
        
        self.log_message("开始收集初始样本...")
        
        while count < self.buffer_num_before_learning_begin:
            # 为每个环境随机选择动作
            actions = [np.random.randint(0, self.agent.action_dim) for _ in range(self.num_envs)]
            
            # 执行动作
            next_states, rewards, dones, _ = self.vec_env.step(actions)
            
            # 存储经验（每个环境）
            for i in range(self.num_envs):
                self.agent.replay_buffer.push(
                    states[i], 
                    actions[i], 
                    rewards[i], 
                    next_states[i], 
                    dones[i]
                )
                count += 1
                
                if count >= self.buffer_num_before_learning_begin:
                    break
            
            # 对于已经结束的环境，重置它们
            for i in range(self.num_envs):
                if dones[i]:
                    # 重置单个环境
                    try:
                        states[i] = self.vec_env.reset_one(i)
                    except:
                        # 如果不支持单独重置，跳过
                        pass
            
            # 更新状态
            states = next_states
        
        self.initialized = True
        self.log_message(f"已收集 {count} 个初始样本用于训练")
    
    def train_one_iteration(self, max_steps=1000):
        """
        训练一个迭代（在所有环境中并行执行）
        
        参数:
            max_steps: 最大步数
            
        返回:
            total_reward: 所有环境的总奖励均值
            avg_loss: 平均损失
            steps: 回合步数
            train_info: 详细训练信息字典
        """
        # 确保已初始化
        if not self.initialized:
            self.init_training()
            
        # 重置所有环境
        states = self.vec_env.reset()
        
        # 初始化统计数据
        total_rewards = np.zeros(self.num_envs)
        all_dones = np.zeros(self.num_envs, dtype=bool)
        steps = 0
        losses = []
        
        # 每个环境的奖励和是否完成
        episode_rewards = np.zeros(self.num_envs)
        episode_done_masks = np.zeros(self.num_envs, dtype=bool)
        
        # 获取有效环境数量和索引
        valid_env_count = self.vec_env.get_valid_env_count()
        valid_env_indices = self.vec_env.get_valid_env_indices()
        
        # 样本收集计数器（用于训练采样比）
        samples_collected = 0
        # 训练次数计数器
        training_count = 0
        
        # 获取batch_size
        batch_size = self.agent.batch_size
        
        # 记录每一步的奖励，用于计算每步平均奖励
        step_rewards = []
        
        while not all(episode_done_masks) and steps < max_steps and valid_env_count > 0:
            # 为每个环境选择动作
            actions = np.array([
                self.agent.select_action(states[i]) 
                for i in range(self.num_envs)
            ])
            
            # 执行动作
            next_states, rewards, dones, infos = self.vec_env.step(actions)
            
            # 累计奖励
            episode_rewards += rewards
            step_rewards.append(np.mean(rewards))
            
            # 存储经验（每个环境）并计算收集的样本数
            samples_in_this_step = 0
            for i in valid_env_indices:
                # 检查环境是否有效
                if 'env_valid' in infos[i] and not infos[i]['env_valid']:
                    continue
                    
                self.agent.replay_buffer.push(
                    states[i], 
                    actions[i], 
                    rewards[i], 
                    next_states[i], 
                    dones[i]
                )
                samples_in_this_step += 1
            
            samples_collected += samples_in_this_step
            
            # 确保经验回放缓冲区中有足够的样本进行训练
            if len(self.agent.replay_buffer) < batch_size:
                # 如果样本不足一个批次，则跳过训练部分
                self.log_message(f"样本数 {len(self.agent.replay_buffer)} 不足一个批次 {batch_size}，等待收集更多样本")
                # 更新状态和完成状态
                states = next_states
                episode_done_masks = np.logical_or(episode_done_masks, dones)
                steps += 1
                continue
            
            # 计算应该进行的训练批次数
            # 训练批次数 = 样本总数 * 训练采样比 / 批次大小
            required_training_batches = int(samples_collected * self.train_sample_ratio / batch_size)
            
            # 如果已完成的训练批次少于需要的批次数，执行训练
            if training_count < required_training_batches:
                # 执行训练，直到达到所需批次数
                training_steps_this_round = required_training_batches - training_count
                
                for _ in range(training_steps_this_round):
                    loss = self.agent.learn()
                    if loss != 0.0:
                        losses.append(loss)
                
                # 更新训练计数
                training_count += training_steps_this_round
                
                if training_steps_this_round > 0:
                    self.log_message(f"执行了 {training_steps_this_round} 次训练，总样本数: {samples_collected}, " 
                                    f"批次大小: {batch_size}, 训练比例: {self.train_sample_ratio}, "
                                    f"总训练次数: {training_count}")
            
            # 更新状态和完成状态
            states = next_states
            episode_done_masks = np.logical_or(episode_done_masks, dones)
            
            # 更新步数
            steps += 1
            
            # 更新有效环境计数
            valid_env_count = self.vec_env.get_valid_env_count()
            valid_env_indices = self.vec_env.get_valid_env_indices()
            
            # 每隔一定步数检查环境状态
            if steps % 100 == 0:
                env_status = self.vec_env.get_env_status_summary()
                self.log_message(f"步数 {steps}: 有效环境 {env_status['valid']}/{env_status['total']} ({env_status['valid_percent']})")
                self.log_message(f"收集样本数: {samples_collected}, 批次大小: {batch_size}, "
                                f"训练比例: {self.train_sample_ratio}, "
                                f"已训练批次数: {training_count}, "
                                f"需要训练批次数: {required_training_batches}")
        
        # 计算平均值
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(losses) if losses else 0.0
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        win_rate = np.mean(episode_rewards > 0) if len(episode_rewards) > 0 else 0.0
        lose_rate = np.mean(episode_rewards < 0) if len(episode_rewards) > 0 else 0.0
        draw_rate = np.mean(episode_rewards == 0) if len(episode_rewards) > 0 else 0.0
        
        # 创建要返回的训练信息字典
        train_info = {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'steps': steps,
            'avg_step_reward': avg_step_reward,
            'win_rate': win_rate,
            'lose_rate': lose_rate,
            'draw_rate': draw_rate,
            'samples_collected': samples_collected,
            'training_count': training_count,
            'buffer_size': len(self.agent.replay_buffer),
            'completed_episodes': np.sum(episode_done_masks)  # 计算完成的有效轨迹数量
        }
        
        return avg_reward, avg_loss, steps, train_info
    
    def evaluate(self):
        """
        评估当前策略（使用单环境）
        
        返回:
            avg_reward: 平均奖励
        """
        total_reward = 0
        
        for ep in range(self.num_evaluate_episode):
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
            path = f"{self.ckpt_path}/policy_net_parallel.pt"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.agent.policy_net.state_dict(), path)
        self.log_message(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        self.agent.policy_net.load_state_dict(torch.load(path))
        self.agent.update_target_network()
        self.log_message(f"模型已从 {path} 加载")


if __name__ == "__main__":
    args = parse_arguments()
    train(args)