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
DQN config.
"""

import mindspore as ms
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from .dqn import DQNActor, DQNLearner, DQNPolicy

learner_params = {'gamma': 0.9, 'lr': 0.001,}
trainer_params = {
    'buffer_num_before_learning_begin': 32,
    'update_target_iter': 100,
    'num_evaluate_episode': 3,
    'ckpt_path': './ckpt',
}

collect_env_params = {'name': 'FightingiceEnv-v0'}
eval_env_params = {'name': 'FightingiceEnv-v0'}

# collect_env_params = {'name': 'CartPole-v0'}
# eval_env_params = {'name': 'CartPole-v0'}

# policy_params = {
#     'epsi_high': 0.9,
#     'epsi_low': 0.01,
#     'decay': 250,
#     'state_space_dim': 0,
#     'action_space_dim': 0,
#     'hidden_size': 100,
# }

policy_params = {
    'epsi_high': 0.9,
    'epsi_low': 0.05,
    'decay': 200,
    'state_space_dim': 144,
    'action_space_dim': 40,
    'hidden_size': 128,
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': DQNActor,
        'policies': ['init_policy', 'collect_policy', 'evaluate_policy'],
    },
    'learner': {
        'number': 1,
        'type': DQNLearner,
        'params': learner_params,
        'networks': ['policy_network', 'target_network']
    },
    'policy_and_network': {
        'type': DQNPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': collect_env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {'number': 1,
                      'type': UniformReplayBuffer,
                      'capacity': 20000,
                      'data_shape': [(144,), (1,), (1,), (144,)],
                      'data_type': [ms.float32, ms.int32, ms.float32, ms.float32],
                      'sample_size': 32},
}