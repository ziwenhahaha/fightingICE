# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
AC config.
"""

from mindspore_rl.environment import GymEnvironment
from .ac import ACPolicyAndNetwork, ACLearner, ACActor

env_params = {'name': 'FightingiceEnv-v0', 'seed': 42}
policy_params = {
    'state_space_dim': 144,
    'action_space_dim': 40,
    'hidden_size': 128,
}
trainer_params = {
    'num_evaluate_episode': 3,
    'ckpt_path': './ckpt',
}
learner_params = {
    'gamma': 0.9,
    'state_space_dim': 4,
    'action_space_dim': 2,
    'alr': 0.001,
    'clr': 0.01,
}
algorithm_config = {
    'actor': {
        'number': 1,
        'type': ACActor,
        'policies': ['collect_policy', 'eval_policy'],
    },
    'learner': {
        'number': 1,
        'type': ACLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net']
    },
    'policy_and_network': {
        'type': ACPolicyAndNetwork,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': env_params
    },
}
