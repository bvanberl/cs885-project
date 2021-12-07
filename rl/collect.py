import os

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy

from rl.dqn import DQN_CNN
from envs import classic_control

def collect_experience_with_policy(env, policy_type, policy_path, experience_dir, n_frames, device='cuda', lr=0.0001, n_steps=10000):

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    if not os.path.exists(experience_dir):
        os.makedirs(experience_dir)

    if policy_type == 'DQN':
        net = DQN_CNN(n_frames, *state_shape, action_shape, device).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        policy = DQNPolicy(net, optim, 0.99, 3, target_update_freq=5000)
    else:
        raise Exception("Unsupported policy type.")
    s_dict = torch.load(policy_path, map_location=device)['model']
    policy.load_state_dict(s_dict)
    policy.eval()
    policy.set_eps(0)   # Set epsilon to 0 for greedy action selection

    buffer = VectorReplayBuffer(total_size=n_steps, buffer_num=1)
    collector = Collector(policy, env, buffer=buffer)
    collector.collect(n_step=n_steps, render=None)
    collector.buffer.save_hdf5(os.path.join(experience_dir, 'buf.hdf5'))

if __name__=='__main__':
    n_frames = 4
    env = classic_control.ClassicControlEnv('CartPole-v0', n_frames=n_frames, seed=10)
    policy_type = 'DQN'
    policy_path = 'results/log/cartpole2/CartPole-v0/dqn/checkpoint.pth'
    experience_dir = 'data/experience/cartpole2/train'
    collect_experience_with_policy(env, policy_type, policy_path, experience_dir, n_frames, n_steps=10000)

