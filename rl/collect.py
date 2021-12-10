import os

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.env import ShmemVectorEnv, SubprocVectorEnv, DummyVectorEnv

from rl.dqn import DQN_CNN
from envs import classic_control

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def create_data_for_causal_vae(data_dir, buffer, obs_len, discrete_actions=False):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    actions = []
    concepts = []
    obs_paths = []
    step_ctr = 0
    for i in tqdm(range(len(buffer))):
        step_ctr += 1
        if step_ctr < obs_len:
            continue
        xp = buffer[i]

        obs_path = os.path.join(data_dir, f'{i}.npy')
        with open(obs_path, 'wb') as f:
            np.save(f, xp.obs)

        obs_paths.append(obs_path)

        actions.append(xp.act)
        concepts.append(xp.info.concepts)

        if xp.done:
            step_ctr = 0

    actions = np.stack(actions, axis=0)
    if discrete_actions:
        actions = actions.astype(np.int)
        actions = np.eye(np.max(actions) + 1)[actions]
    if len(actions.shape) < 2:
        actions = np.expand_dims(actions, axis=-1)
    concepts = np.stack(concepts, axis=0)
    xp_df = pd.DataFrame(dict({'Observation': obs_paths},
                              **{f'Action_{a}': actions[:, a] for a in range(actions.shape[1])},
                              **{f'Concept_{c}': concepts[:, c] for c in range(concepts.shape[1])}))
    xp_df.to_csv(os.path.join(data_dir, 'xp_df.csv'), index=False)
    return

def collect_experience_with_policy(env_name, n_envs, policy_type, policy_path, n_frames, device='cuda', lr=0.0001, n_steps=10000):

    envs = DummyVectorEnv(
        [lambda: classic_control.ClassicControlEnv(env_name, n_frames=n_frames, seed=i) for i in range(n_envs)]
    )
    env = classic_control.ClassicControlEnv(env_name, n_frames=n_frames, seed=0)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    if policy_type == 'DQN':
        model_cfg = cfg['RL'][env_name.upper()]['DQN']
        net = DQN_CNN(model_cfg, n_frames, *state_shape, action_shape, device).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        policy = DQNPolicy(net, optim, 0.99, 3, target_update_freq=5000)
    else:
        raise Exception("Unsupported policy type.")
    s_dict = torch.load(policy_path, map_location=device)['model']
    policy.load_state_dict(s_dict)
    policy.eval()
    policy.set_eps(0)   # Set epsilon to 0 for greedy action selection

    buffer = VectorReplayBuffer(total_size=n_steps, buffer_num=n_envs)
    collector = Collector(policy, envs, buffer=buffer)
    collector.collect(n_step=n_steps, render=None)
    #collector.buffer.save_hdf5(os.path.join(experience_dir, 'buf.hdf5'))
    return collector.buffer

def generate_xp_dataset(env_name, n_envs, policy_type, policy_path, data_dir, n_frames, device='cuda', lr=0.0001, n_steps=10000):
    buffer = collect_experience_with_policy(env_name, n_envs, policy_type, policy_path, n_frames,
                                            device=device, lr=lr, n_steps=n_steps)
    if policy_type == 'DQN':
        discrete_actions = True
    else:
        discrete_actions = False
    create_data_for_causal_vae(data_dir, buffer, n_frames, discrete_actions=discrete_actions)

if __name__=='__main__':
    n_frames = 4
    env_name = 'CartPole-v0'
    n_envs = 5
    policy_type = 'DQN'
    policy_path = 'results/log/rl/cartpole2/CartPole-v0/dqn/checkpoint.pth'
    experience_dir = 'data/experience/cartpole2/train'
    data_dir = 'B:/Datasets/School/CausalVAE/cartpole2/train'
    #collect_experience_with_policy(env_name, n_envs, policy_type, policy_path, experience_dir, n_frames, n_steps=50000)
    generate_xp_dataset(env_name, n_envs, policy_type, policy_path, data_dir, n_frames, n_steps=25000)

