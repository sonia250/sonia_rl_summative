"""Watch a trained agent in the Court Motion Triage environment (GUI + terminal output).

Examples (from the project root):
    uv run play.py                      # best PPO agent, 1 episode
    uv run play.py --algo dqn           # best DQN agent
    uv run play.py --algo ppo --fps 5 --episodes 2
    uv run play.py --random             # random agent for comparison
"""
import argparse
import os
import sys

import torch
from torch import nn
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import ACTION_NAMES, CourtMotionEnv


def build_net(hidden, activation):
    return nn.Sequential(nn.Linear(29, hidden), activation(),
                         nn.Linear(hidden, hidden), activation(),
                         nn.Linear(hidden, 8))


def load_agent(algo):
    """Returns (net, is_dqn). net is None if the model file is missing."""
    folder = "dqn" if algo == "dqn" else "pg"
    path = os.path.join(ROOT, "models", folder, f"best_{algo}.pt")
    if not os.path.exists(path):
        print(f"[!] Model file not found: {path}  -> using a random agent")
        return None, algo == "dqn"
    ckpt = torch.load(path, map_location="cpu")
    net = build_net(ckpt["hidden"], nn.ReLU if algo == "dqn" else nn.Tanh)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, algo == "dqn"


def choose_action(net, obs, is_dqn, env, greedy):
    if net is None:
        return env.action_space.sample()
    with torch.no_grad():
        logits = net(torch.as_tensor(obs, dtype=torch.float32))
    if is_dqn or greedy:
        return int(logits.argmax())
    return int(Categorical(logits=logits).sample())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], default="ppo")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=6, help="speed of the animation")
    p.add_argument("--seed", type=int, default=10_000, help="first episode seed (unseen during training)")
    p.add_argument("--greedy", action="store_true", help="policy-gradient agents: pick the most likely action")
    p.add_argument("--random", action="store_true", help="use a random agent instead")
    p.add_argument("--no-wait", action="store_true", help="do not wait for Enter at the end")
    args = p.parse_args()

    net, is_dqn = (None, False) if args.random else load_agent(args.algo)
    label = "RANDOM" if net is None else args.algo.upper()

    CourtMotionEnv.metadata["render_fps"] = args.fps
    env = CourtMotionEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\n===== Episode {ep + 1} | agent: {label} | seed {args.seed + ep} =====")
        total, done, step = 0.0, False, 0
        while not done:
            action = choose_action(net, obs, is_dqn, env, args.greedy)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            step += 1
            print(f"step {step:>4} | day {info['day']:>2} | hours left {info['hours_left']:>4.1f} | "
                  f"{ACTION_NAMES[action]:<28} | reward {reward:+6.1f} | backlog {info['backlog']:>2} | "
                  f"resolved {info['resolved']:>3} | overdue {info['overdue_total']:>3}")
            done = terminated or truncated
        print(f"\n--- Episode {ep + 1} finished ---")
        print(f"Total reward: {total:.1f} | resolved: {info['resolved']} | "
              f"overdue: {info['overdue_total']} | final backlog: {info['backlog']} | "
              f"invalid actions: {info['invalid_actions']}")

    if not args.no_wait:
        input("\nPress Enter to close the window...")
    env.close()


if __name__ == "__main__":
    main()
