import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import gym
import sys
import os

# 添加myenv到sys.path以便正确导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
myenv_dir = os.path.join(parent_dir, 'myenv')
if (myenv_dir not in sys.path):
    sys.path.append(myenv_dir)

# 直接导入FightingiceEnv
from myenv.fightingice_env import FightingiceEnv

def worker(remote, parent_remote, env_fn):
    """
    子进程worker，在单独的进程中运行环境
    
    Args:
        remote: 子进程使用的管道端
        parent_remote: 父进程使用的管道端(会被关闭)
        env_fn: 创建环境的函数
    """
    # 关闭父进程的管道端
    parent_remote.close()
    
    # 创建环境
    env = env_fn()
    
    # 持续接收指令并执行
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # 执行环境步骤
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                # 重置环境
                if isinstance(data, dict):
                    obs = env.reset(**data)
                else:
                    obs = env.reset(data)
                remote.send(obs)
                
            elif cmd == 'close':
                # 关闭环境
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                # 获取环境的观察和动作空间
                remote.send((env.observation_space, env.action_space))
                
            else:
                # 未知命令
                raise NotImplementedError(f"Command {cmd} not implemented")
                
        except EOFError:
            # 连接关闭
            break


def worker_with_logging(remote, parent_remote, env_fn, log_path_stdout, log_path_stderr):
    """
    带日志记录功能的子进程worker，在单独的进程中运行环境
    
    Args:
        remote: 子进程使用的管道端
        parent_remote: 父进程使用的管道端(会被关闭)
        env_fn: 创建环境的函数
        log_path_stdout: 标准输出日志文件路径
        log_path_stderr: 标准错误日志文件路径
    """
    try:
        # 关闭父进程的管道端
        parent_remote.close()
        
        # 重定向标准输出和标准错误到日志文件
        with open(log_path_stdout, 'w', buffering=1) as stdout_file, \
             open(log_path_stderr, 'w', buffering=1) as stderr_file:
            
            # 保存原始标准输出和标准错误
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # 重定向输出
            sys.stdout = stdout_file
            sys.stderr = stderr_file
            
            # 记录进程启动信息
            proc_id = mp.current_process().name
            print(f"进程 {proc_id} 启动，PID: {os.getpid()}")
            print(f"标准输出已重定向至: {log_path_stdout}")
            sys.stderr.write(f"标准错误已重定向至: {log_path_stderr}\n")
            
            try:
                # 创建环境
                env = env_fn()
                
                # 持续接收指令并执行
                while True:
                    try:
                        cmd, data = remote.recv()
                        
                        if cmd == 'step':
                            # 执行环境步骤
                            obs, reward, done, info = env.step(data)
                            remote.send((obs, reward, done, info))
                            
                        elif cmd == 'reset':
                            # 重置环境
                            print(f"重置环境...")
                            if isinstance(data, dict):
                                obs = env.reset(**data)
                            else:
                                obs = env.reset(data)
                            remote.send(obs)
                            
                        elif cmd == 'close':
                            # 关闭环境
                            print(f"正在关闭环境...")
                            env.close()
                            remote.close()
                            break
                            
                        elif cmd == 'get_spaces':
                            # 获取环境的观察和动作空间
                            spaces = (env.observation_space, env.action_space)
                            print(f"环境观察空间: {spaces[0]}, 动作空间: {spaces[1]}")
                            remote.send(spaces)
                            
                        else:
                            # 未知命令
                            raise NotImplementedError(f"Command {cmd} not implemented")
                            
                    except EOFError:
                        # 连接关闭
                        print(f"管道连接关闭")
                        break
                        
            except Exception as e:
                # 记录任何异常
                import traceback
                print(f"环境执行过程中发生异常: {str(e)}")
                traceback.print_exc(file=sys.stderr)
                # 尝试关闭连接
                try:
                    remote.close()
                except:
                    pass
                    
            finally:
                # 恢复标准输出和标准错误
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    
    except Exception as e:
        # 捕获在重定向过程中可能发生的任何异常
        import traceback
        print(f"进程初始化过程中发生异常: {str(e)}")
        traceback.print_exc()


class SubprocVecEnv:
    """
    通过子进程实现的并行环境
    
    使用子进程并行运行多个环境，通过管道进行通信
    """
    
    def __init__(self, env_fns):
        """
        Args:
            env_fns: 一个环境创建函数的列表，每个函数无参数并返回一个gym环境
        """
        self.num_envs = len(env_fns)
        
        # 为每个环境创建一对管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        # 启动子进程
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            # 启动进程并将函数和管道传递给进程
            process = mp.Process(target=worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            # 关闭主进程中的工作端管道(worker中已经使用)
            work_remote.close()
        
        # 获取第一个环境的空间信息
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
    
    def step(self, actions):
        """
        并行执行所有环境的一个步骤
        
        Args:
            actions: 动作列表，对应每个环境
            
        Returns:
            obs: 观察值列表
            rews: 奖励列表 
            dones: 完成状态列表
            infos: 信息字典列表
        """
        # 发送所有动作到子进程
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        # 接收所有结果
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        
        return np.stack(obs), np.array(rews), np.array(dones), infos
    
    def reset(self, **kwargs):
        """
        重置所有环境
        
        Returns:
            观察值列表
        """
        # 发送重置命令到所有子进程
        for remote in self.remotes:
            try:
                remote.send(('reset', kwargs))
            except Exception as e:
                print(f"重置环境时发生错误: {e}")
                break
            
        # 接收观察值
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)
        
    def close(self):
        """
        关闭所有环境和子进程
        """
        # 发送关闭命令到所有子进程
        for remote in self.remotes:
            remote.send(('close', None))
            
        # 等待所有进程结束
        for process in self.processes:
            process.join()
            
    def reset_one(self, index, **kwargs):
        """
        重置单个环境
        
        Args:
            index: 环境索引
            
        Returns:
            单个环境的观察值
        """
        self.remotes[index].send(('reset', kwargs))
        return self.remotes[index].recv()
    
    def step_one(self, index, action):
        """
        在单个环境中执行一个步骤
        
        Args:
            index: 环境索引
            action: 动作
            
        Returns:
            单个环境的结果(obs, reward, done, info)
        """
        self.remotes[index].send(('step', action))
        return self.remotes[index].recv()


class SubprocVecEnvWithLogging(SubprocVecEnv):
    """
    带日志记录功能的并行环境
    
    每个子进程的标准输出和标准错误会被重定向到单独的日志文件
    增加了错误处理机制，当某个环境出错时，可以将其标记为无效并继续处理其他环境
    """
    
    def __init__(self, env_fns, log_dir="logs", timeout=30):
        """
        Args:
            env_fns: 一个环境创建函数的列表
            log_dir: 日志文件存放的目录
            timeout: 等待环境响应的超时时间（秒）
        """
        self.num_envs = len(env_fns)
        self.log_dir = log_dir
        self.timeout = timeout
        
        # 环境有效性掩码，初始时所有环境都有效
        self.env_isValids = np.ones(self.num_envs, dtype=bool)
        # 记录每个环境运行状态
        self.env_status = ['正常' for _ in range(self.num_envs)]
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 为每个环境创建一对管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        # 启动子进程
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            # 为每个进程创建唯一的日志文件路径
            log_path_stdout = os.path.join(self.log_dir, f"env_{i}_stdout.log")
            log_path_stderr = os.path.join(self.log_dir, f"env_{i}_stderr.log")
            
            # 启动带日志功能的进程
            process = mp.Process(
                target=worker_with_logging, 
                args=(work_remote, remote, env_fn, log_path_stdout, log_path_stderr),
                daemon=True,
                name=f"EnvWorker-{i}"
            )
            process.start()
            self.processes.append(process)
            # 关闭主进程中的工作端管道
            work_remote.close()
        
        # 获取第一个环境的空间信息
        try:
            self.remotes[0].send(('get_spaces', None))
            if self.remotes[0].poll(self.timeout):
                self.observation_space, self.action_space = self.remotes[0].recv()
            else:
                # 超时，将使用默认空间
                print(f"警告：获取环境空间信息超时，使用默认空间")
                self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(144,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(40)
        except Exception as e:
            print(f"获取环境空间时出错: {e}，使用默认空间")
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(144,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(40)
        
        print(f"创建了 {self.num_envs} 个环境，日志输出到 {self.log_dir} 目录")
    
    def step(self, actions):
        """
        并行执行所有环境的一个步骤，添加了错误处理
        
        Args:
            actions: 动作列表，对应每个环境
            
        Returns:
            obs: 观察值列表
            rews: 奖励列表 
            dones: 完成状态列表
            infos: 信息字典列表
        """
        # 为无效环境创建默认值
        default_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        default_rew = 0.0
        default_done = True  # 标记为已完成
        default_info = {"error": "环境已失效"}
        
        results = []
        
        # 循环处理每个环境
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            # 跳过已标记为无效的环境
            if not self.env_isValids[i]:
                results.append((default_obs, default_rew, default_done, default_info))
                continue
                
            try:
                # 向子进程发送动作
                remote.send(('step', action))
                
                # 等待结果，设置超时
                if remote.poll(self.timeout):
                    obs, rew, done, info = remote.recv()
                    results.append((obs, rew, done, info))
                else:
                    print(f"环境 {i} 执行超时，标记为无效")
                    self.env_isValids[i] = False
                    self.env_status[i] = '超时'
                    results.append((default_obs, default_rew, default_done, {"error": "环境响应超时"}))
            
            except Exception as e:
                print(f"环境 {i} 执行出错: {str(e)}，标记为无效")
                self.env_isValids[i] = False
                self.env_status[i] = f'错误: {str(e)}'
                results.append((default_obs, default_rew, default_done, {"error": str(e)}))
        
        # 分解结果
        obs, rews, dones, infos = zip(*results)
        
        # 为infos中每个元素添加环境有效性标记（处理列表类型的info）
        processed_infos = []
        for i, (info, mask) in enumerate(zip(infos, self.env_isValids)):
            # 创建一个新的字典
            info_dict = {}
            
            # 如果是字典类型，直接复制
            if isinstance(info, dict):
                info_dict.update(info)
            # 如果是列表类型，将其转换为字典
            elif isinstance(info, list):
                info_dict["action_list"] = info  # 将列表存储为action_list键
            # 其他类型，记录类型信息
            elif info is not None:
                info_dict["original_info_type"] = str(type(info))
                info_dict["original_info"] = str(info)
            
            # 添加环境有效性标记
            info_dict["env_valid"] = mask
            processed_infos.append(info_dict)
        
        return np.array(obs), np.array(rews), np.array(dones), processed_infos
    
    def reset(self, **kwargs):
        """
        重置所有有效的环境，添加了错误处理
        
        Returns:
            观察值列表
        """
        # 为无效环境创建默认观察值
        default_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        observations = []
        
        # 循环处理每个环境
        for i, remote in enumerate(self.remotes):
            # 跳过已标记为无效的环境
            if not self.env_isValids[i]:
                observations.append(default_obs)
                continue
                
            try:
                # 发送重置命令
                remote.send(('reset', kwargs))
                
                # 等待结果，设置超时
                if remote.poll(self.timeout):
                    obs = remote.recv()
                    observations.append(obs)
                else:
                    print(f"环境 {i} 重置超时，标记为无效")
                    self.env_isValids[i] = False
                    self.env_status[i] = '重置超时'
                    observations.append(default_obs)
            
            except Exception as e:
                print(f"环境 {i} 重置出错: {str(e)}，标记为无效")
                self.env_isValids[i] = False
                self.env_status[i] = f'重置错误: {str(e)}'
                observations.append(default_obs)
        
        return np.array(observations)
    
    def get_valid_env_count(self):
        """
        获取当前有效环境的数量
        
        Returns:
            有效环境数量
        """
        return np.sum(self.env_isValids)
    
    def get_valid_env_indices(self):
        """
        获取当前有效环境的索引
        
        Returns:
            有效环境索引列表
        """
        return np.where(self.env_isValids)[0]
    
    def get_env_status_summary(self):
        """
        获取环境状态摘要
        
        Returns:
            状态摘要字典
        """
        return {
            "total": self.num_envs,
            "valid": int(self.get_valid_env_count()),
            "valid_percent": f"{100 * self.get_valid_env_count() / self.num_envs:.1f}%",
            "status": self.env_status
        }
    
    def close(self):
        """
        关闭所有环境和子进程
        """
        # 发送关闭命令到所有子进程
        for i, remote in enumerate(self.remotes):
            try:
                if self.env_isValids[i]:  # 只关闭有效环境
                    remote.send(('close', None))
            except:
                pass
            
        # 等待所有进程结束
        for process in self.processes:
            process.join(timeout=2)  # 设置一个超时避免卡住
            if process.is_alive():
                print(f"警告：进程 {process.name} 未能正常结束，强制终止")
                process.terminate()


def make_env(env_name, **kwargs):
    """
    创建环境的工厂函数
    
    Args:
        env_name: 环境名称，目前仅支持"FightingiceEnv"
        **kwargs: 传递给环境构造函数的参数
        
    Returns:
        创建并返回环境的函数
    """
    def _make():
        # 这里使用不同的port参数，避免端口冲突
        if 'port' not in kwargs:
            try:
                import port_for
                kwargs['port'] = port_for.select_random()  # 随机选择一个端口
            except ImportError:
                # 如果没有port_for库，使用随机端口
                import random
                kwargs['port'] = random.randint(4000, 8000)
        
        # 直接创建FightingiceEnv实例，不再使用gym.make
        if env_name.startswith('FightingiceEnv'):
            return FightingiceEnv(**kwargs)
        else:
            # 如果不是FightingiceEnv，尝试使用gym.make（兼容其他环境）
            return gym.make(env_name, **kwargs)
    return _make


def make_vec_env(env_name, n_envs=4, **env_kwargs):
    """
    创建并行环境的便捷函数
    
    Args:
        env_name: 环境名称，比如"FightingiceEnv-v0"
        n_envs: 并行环境数量
        **env_kwargs: 传递给环境的参数
        
    Returns:
        并行环境实例
    """
    print(f"创建 {n_envs} 个 {env_name} 环境...")
    env_fns = [make_env(env_name, **env_kwargs) for _ in range(n_envs)]
    return SubprocVecEnv(env_fns)


def make_vec_env_with_logging(env_name, n_envs=4, log_dir="logs", **env_kwargs):
    """
    创建带日志记录功能的并行环境
    
    Args:
        env_name: 环境名称，比如"FightingiceEnv-v0" 
        n_envs: 并行环境数量
        log_dir: 日志文件存放目录
        **env_kwargs: 传递给环境的参数
        
    Returns: 
        带日志功能的并行环境实例
    """
    print(f"创建 {n_envs} 个 {env_name} 环境，日志输出到 {log_dir}...")
    env_fns = [make_env(env_name, **env_kwargs) for _ in range(n_envs)]
    return SubprocVecEnvWithLogging(env_fns, log_dir=log_dir)