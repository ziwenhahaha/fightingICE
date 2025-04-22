import numpy as np
import time
import os
import sys
from src.vec_env import SubprocVecEnv  # 修改导入为我们实现的VecEnv类

# 简单的随机策略类
class RandomPolicy:
    def __init__(self, action_dim=40):
        self.action_dim = action_dim
        
    def select_actions(self, observations):
        # 为每个观察选择一个随机动作
        batch_size = len(observations)
        return np.random.randint(0, self.action_dim, size=batch_size)

def test_vector_env(num_envs=4, num_steps=100):
    print(f"开始测试向量环境 (并行环境数: {num_envs})")
    
    try:
        # 创建并行环境
        print("创建向量环境...")
        # 直接实例化VecEnv类
        vec_env = SubprocVecEnv(num_envs=num_envs, p2_name="MctsAi", start_port=4000)
        
        # 创建随机策略
        policy = RandomPolicy()
        
        # 重置所有环境
        print("重置环境...")
        start_time = time.time()
        obs = vec_env.reset()
        print(f"重置完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"观察形状: {obs.shape}")
        
        # 收集轨迹
        print(f"开始收集{num_steps}步的轨迹...")
        trajectories = [[] for _ in range(num_envs)]
        total_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=int)
        
        start_time = time.time()
        for step in range(num_steps):
            # 使用策略为每个观察选择动作
            actions = policy.select_actions(obs)
            
            # 并行执行动作
            next_obs, rewards, dones, infos = vec_env.step(actions)
            
            # 收集数据
            for i in range(num_envs):
                trajectories[i].append({
                    'observation': obs[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_observation': next_obs[i],
                    'done': dones[i]
                })
                total_rewards[i] += rewards[i]
                episode_lengths[i] += 1
            
            # 打印进度
            if (step + 1) % 10 == 0:
                print(f"已完成 {step + 1}/{num_steps} 步，耗时: {time.time() - start_time:.2f}秒")
            
            obs = next_obs
        
        # 打印统计信息
        print("\n测试完成！")
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        print(f"平均每步耗时: {(time.time() - start_time) / num_steps:.4f}秒")
        print(f"每个环境的总奖励: {total_rewards}")
        print(f"每个环境的轨迹长度: {episode_lengths}")
        
        # 关闭环境
        vec_env.close()
        print("环境已关闭")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 从命令行参数获取并行环境数和步数
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    test_vector_env(num_envs, num_steps)