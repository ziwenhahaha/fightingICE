"""
并行环境使用示例
"""
import numpy as np
import time
import argparse
import os
import sys
from src.vec_env import make_vec_env_with_logging

def main():
    parser = argparse.ArgumentParser(description='并行环境示例')
    parser.add_argument('--num_envs', type=int, default=4, help='并行环境数量')
    parser.add_argument('--episodes', type=int, default=2, help='运行的回合数')
    parser.add_argument('--steps_per_episode', type=int, default=100, help='每个回合的步数')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志输出目录')
    parser.add_argument('--timeout', type=int, default=30, help='环境响应的超时时间(秒)')
    args = parser.parse_args()
    
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"启动 {args.num_envs} 个并行环境...")
    
    # 创建并行环境 - 使用带日志和错误处理功能的函数
    env = make_vec_env_with_logging(
        'FightingiceEnv', 
        n_envs=args.num_envs,
        log_dir=args.log_dir,
        timeout=args.timeout,  # 设置超时时间
        # 可以传入其他参数，例如:
        p2="MctsAi" # 指定对手
    )
    
    try:
        # 打印环境信息
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 重置所有环境
        print("重置环境...")
        obs = env.reset()
        
        # 检查有效环境数量
        valid_count = env.get_valid_env_count()
        print(f"重置后的有效环境数量: {valid_count}/{args.num_envs}")
        
        if valid_count == 0:
            print("错误: 所有环境都无效，无法继续")
            return
        
        total_reward = 0
        episode_rewards = []
        
        start_time = time.time()
        
        # 运行多个回合
        for episode in range(args.episodes):
            print(f"\n回合 {episode+1}/{args.episodes}")
            episode_reward = np.zeros(args.num_envs)
            
            # 重置环境
            try:
               obs = env.reset()
               # 更新有效环境信息
               valid_count = env.get_valid_env_count()
               valid_indices = env.get_valid_env_indices()
               print(f"重置后的有效环境数量: {valid_count}/{args.num_envs}")
               
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
                
            # 在每个环境中执行多个步骤
            for step in range(args.steps_per_episode):
                # 为所有环境选择动作（无效环境的动作不会被使用）
                actions = [env.action_space.sample() for _ in range(args.num_envs)]
                # 执行动作
                try:
                    next_obs, rewards, dones, infos = env.step(actions)
                    
                    # 更新有效环境信息（环境可能在step过程中变为无效）
                    valid_count = env.get_valid_env_count()
                    valid_indices = env.get_valid_env_indices()
                    
                    if valid_count == 0:
                        print(f"警告: 所有环境都已无效，结束本回合")
                        break
                    
                except Exception as e:
                    print(f"执行动作时发生错误: {e}")
                    break
                    
                # 只累加有效环境的奖励
                for i, info in enumerate(infos):
                    if info.get("env_valid", False):
                        episode_reward[i] += rewards[i]
                
                # 如果游戏结束，打印信息但继续使用该环境（已被重置）
                for i in valid_indices:
                    if dones[i]:
                        print(f"环境 {i} 完成，奖励: {episode_reward[i]:.2f}")
                
                obs = next_obs
                
                # 每10步显示进度
                if step % 10 == 0:
                    # 只考虑有效环境的平均奖励
                    valid_rewards = [episode_reward[i] for i in valid_indices]
                    mean_reward = np.mean(valid_rewards) if valid_rewards else 0
                    
                    print(f"回合 {episode+1}, 步骤 {step}/{args.steps_per_episode}, "
                          f"有效环境: {valid_count}/{args.num_envs}, "
                          f"平均奖励: {mean_reward:.2f}")
            
            # 只记录有效环境的总奖励
            valid_rewards = [episode_reward[i] for i in valid_indices]
            if valid_rewards:
                mean_reward = np.mean(valid_rewards)
                total_reward += sum(valid_rewards)
                episode_rewards.append(mean_reward)
                print(f"回合 {episode+1} 平均奖励: {mean_reward:.2f} (有效环境数: {valid_count})")
            else:
                print(f"回合 {episode+1} 无有效数据")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("\n========= 结果汇总 =========")
        print(f"总计回合数: {args.episodes}")
        if episode_rewards:
            print(f"平均每回合奖励: {np.mean(episode_rewards):.2f}")
        else:
            print("没有收集到有效数据")
        print(f"总运行时间: {elapsed_time:.2f}秒")
        if episode_rewards:
            print(f"每回合平均时间: {elapsed_time/len(episode_rewards):.2f}秒")
        
        # 显示每个环境的最终状态
        status_summary = env.get_env_status_summary()
        print("\n环境状态汇总:")
        print(f"总环境数: {status_summary['total']}")
        print(f"有效环境数: {status_summary['valid']} ({status_summary['valid_percent']})")
        print("各环境状态:")
        for i, status in enumerate(status_summary['status']):
            print(f"环境 {i}: {status}")
    
    finally:
        # 确保环境正确关闭
        env.close()
        print("所有环境已关闭")

if __name__ == "__main__":
    main()