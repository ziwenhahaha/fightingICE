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
import argparse
import os
from src import config
from src.dqn_trainer import DQNTrainer
from mindspore import context
from mindspore import dtype as mstype
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DQN')
parser.add_argument('--episode', type=int, default=1000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the dqn example(Default: Auto).')
parser.add_argument('--parallel', type=bool, default=False, help='Enable parallel training.')
parser.add_argument('--device_num', type=int, default=1, help='Number of devices for parallel training.')
parser.add_argument('--distribute', type=bool, default=False, help='Run distribute training.')
parser.add_argument('--rank_id', type=int, default=0, help='Rank id for distributed training.')
options, _ = parser.parse_known_args()

def train(episode=options.episode):
    """start to train dqn algorithm"""
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)

    # 配置并行训练参数
    if options.parallel or options.distribute:
        device_num = options.device_num
        rank_id = options.rank_id
        
        if options.distribute:
            # 分布式训练设置
            context.set_auto_parallel_context(device_num=device_num, 
                                             parallel_mode=context.ParallelMode.DATA_PARALLEL, 
                                             gradients_mean=True)
            context.set_context(mode=context.GRAPH_MODE)
            # 更新算法配置中的actor和learner数量
            config.algorithm_config['actor']['number'] = device_num
            config.algorithm_config['learner']['number'] = device_num
        else:
            # 本地并行训练
            context.set_context(mode=context.GRAPH_MODE, device_id=rank_id)
    else:
        context.set_context(mode=context.GRAPH_MODE)

    compute_type = mstype.float16 if context.get_context('device_target') in ['Ascend'] else mstype.float32
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type

    dqn_session = Session(config.algorithm_config)
    loss_cb = LossCallback()
    ckpt_cb = CheckpointCallback(50, config.trainer_params['ckpt_path'])
    eval_cb = EvaluateCallback(10)
    cbs = [loss_cb, ckpt_cb, eval_cb]
    dqn_session.run(class_type=DQNTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)

if __name__ == "__main__":
    train()
