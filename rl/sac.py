import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from envs.classic_control import ClassicControlEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="MountainCarContinuous-v0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='results/log/rl/mcc1')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--n-frames', type=str, default=4)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    return parser.parse_args()


class Actor_CNN(nn.Module):

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)

    def forward(self, x: Union[np.ndarray, torch.Tensor], state: Optional[Any] = None, info: Dict[str, Any] = {}) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 1)
        return self.net(x), state

class Critic_CNN(nn.Module):

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)

    def forward(self, x: Union[np.ndarray, torch.Tensor], state: Optional[Any] = None, info: Dict[str, Any] = {}) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 1)
        return self.net(x), state

def make_custom_control_env(args, seed):
    return ClassicControlEnv(args.task, seed=seed, n_frames=args.n_frames)

def train_sac(args=get_args()):
    env = make_custom_control_env(args, args.seed)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Actions space:", env.action_space)

    train_envs = DummyVectorEnv(
        [lambda: make_custom_control_env(args, args.seed + i) for i in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: make_custom_control_env(args, args.seed + args.training_num + j) for j in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #train_envs.seed(args.seed)
    #test_envs.seed(args.seed)

    # model
    #net_a = Actor_CNN(args.n_frames, *args.state_shape, args.action_shape, args.device, features_only=True).to(args.device)
    net_a = Net([args.n_frames, args.state_shape[0], args.state_shape[1]], hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net_a, args.action_shape, max_action=args.max_action, device=args.device, unbounded=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    #net_c1 = Critic_CNN(args.n_frames, *args.state_shape, args.action_shape, args.device, features_only=True).to(args.device)
    net_c1 = Net([args.n_frames, args.state_shape[0], args.state_shape[1]], args.action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    #net_c2 = Critic_CNN(args.n_frames, *args.state_shape, args.action_shape, args.device, features_only=True).to(args.device)
    net_c2 = Net([args.n_frames, args.state_shape[0], args.state_shape[1]], args.action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)), exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    train_sac()