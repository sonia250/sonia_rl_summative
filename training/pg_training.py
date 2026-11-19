"""Policy-gradient training with TorchRL: REINFORCE, PPO and A2C.

Run from the project root:
    uv run python training/pg_training.py --algo ppo --smoke      # 1-minute test
    uv run python training/pg_training.py --algo ppo --run 1      # one run
    uv run python training/pg_training.py --algo ppo --all        # all 12 runs

--algo is one of: reinforce, ppo, a2c

Note: TorchRL has no ready-made REINFORCE loss, so REINFORCE is written by hand
(TorchRL collector + ProbabilisticActor, own Monte-Carlo return and loss).
PPO uses ClipPPOLoss + GAE; A2C uses A2CLoss + GAE.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tensordict.nn import TensorDictModule

try:  # TorchRL >= 0.11 renamed the collector
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymWrapper
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from environment.custom_env import CourtMotionEnv

MODEL_DIR = os.path.join(ROOT, "models", "pg")

# ---------------------------------------------------------------------------
# 12 hyperparameter combinations per algorithm (run 1 = baseline).
# All algorithms use the same total number of frames (60,000) as DQN,
# so the comparison is fair.
# ---------------------------------------------------------------------------
COMMON = dict(total_frames=60_000, hidden=128, grad_clip=0.5)

BASE = {
    "reinforce": dict(lr=1e-3, gamma=0.99, frames_per_batch=1000, entropy_coeff=0.01,
                      normalize_returns=True),
    "ppo": dict(lr=3e-4, gamma=0.99, lmbda=0.95, frames_per_batch=1000, clip_epsilon=0.2,
                entropy_coeff=0.01, epochs=8, minibatch=128, critic_coeff=1.0),
    "a2c": dict(lr=1e-3, gamma=0.99, lmbda=0.95, frames_per_batch=200, entropy_coeff=0.01,
                critic_coeff=1.0),
}
CHANGES = {
    "reinforce": [
        {},                                                        # 1 baseline
        {"lr": 3e-4},                                              # 2 lower lr
        {"lr": 3e-3},                                              # 3 higher lr
        {"gamma": 0.95},                                           # 4 short horizon
        {"gamma": 0.999},                                          # 5 long horizon
        {"entropy_coeff": 0.0},                                    # 6 no entropy bonus
        {"entropy_coeff": 0.05},                                   # 7 strong entropy bonus
        {"frames_per_batch": 2000},                                # 8 bigger batch
        {"frames_per_batch": 500},                                 # 9 smaller batch
        {"normalize_returns": False, "lr": 1e-4},                  # 10 raw returns
        {"hidden": 256},                                           # 11 bigger network
        {"gamma": 0.95, "frames_per_batch": 2000, "lr": 3e-4},     # 12 combination
    ],
    "ppo": [
        {},                                                        # 1 baseline
        {"lr": 1e-3},                                              # 2 higher lr
        {"lr": 1e-4},                                              # 3 lower lr
        {"clip_epsilon": 0.1},                                     # 4 tighter clipping
        {"clip_epsilon": 0.3},                                     # 5 looser clipping
        {"entropy_coeff": 0.0},                                    # 6 no entropy bonus
        {"entropy_coeff": 0.05},                                   # 7 strong entropy bonus
        {"lmbda": 0.9},                                            # 8 lower GAE lambda
        {"lmbda": 0.99},                                           # 9 higher GAE lambda
        {"epochs": 4},                                             # 10 fewer epochs
        {"frames_per_batch": 2000, "minibatch": 256},              # 11 longer rollouts
        {"gamma": 0.95, "hidden": 256},                            # 12 short horizon, big net
    ],
    "a2c": [
        {},                                                        # 1 baseline
        {"lr": 3e-4},                                              # 2 lower lr
        {"lr": 3e-3},                                              # 3 higher lr
        {"frames_per_batch": 100},                                 # 4 shorter rollouts
        {"frames_per_batch": 500},                                 # 5 longer rollouts
        {"entropy_coeff": 0.0},                                    # 6 no entropy bonus
        {"entropy_coeff": 0.05},                                   # 7 strong entropy bonus
        {"gamma": 0.95},                                           # 8 short horizon
        {"lmbda": 0.8},                                            # 9 lower GAE lambda
        {"critic_coeff": 0.5},                                     # 10 weaker critic loss
        {"hidden": 256},                                           # 11 bigger network
        {"lmbda": 0.99, "lr": 5e-4, "frames_per_batch": 400},      # 12 combination
    ],
}


def get_runs(algo):
    return [{**COMMON, **BASE[algo], **c} for c in CHANGES[algo]]


def make_env():
    return GymWrapper(CourtMotionEnv(), categorical_action_encoding=True)


def mlp(n_in, n_out, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def discounted_returns(rewards, dones, gamma):
    """Reward-to-go for a batch of frames; resets at episode ends.
    Unfinished episode at the end of the batch is cut (bootstrap = 0)."""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
        running = rewards[t] + gamma * running
        G[t] = running
    return torch.as_tensor(G)


def evaluate(actor_net, n_episodes=20, seed0=10_000):
    """Greedy policy on NEW random seeds (generalization check)."""
    rets, overdue, backlog, resolved = [], [], [], []
    env = CourtMotionEnv()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                logits = actor_net(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, term, trunc, info = env.step(int(logits.argmax()))
            total += r
            done = term or trunc
        rets.append(total)
        overdue.append(info["overdue_total"])
        backlog.append(info["backlog"])
        resolved.append(info["resolved"])
    return dict(eval_mean=float(np.mean(rets)), eval_std=float(np.std(rets)),
                eval_overdue=float(np.mean(overdue)), eval_backlog=float(np.mean(backlog)),
                eval_resolved=float(np.mean(resolved)))


def train(algo, run_id, hp):
    log_dir = os.path.join(ROOT, "logs", algo)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env()
    actor_net = mlp(29, 8, hp["hidden"])
    policy = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], distribution_class=Categorical,
        spec=env.action_spec, return_log_prob=True,
    )

    adv_module, loss_module, params = None, None, list(actor_net.parameters())
    if algo in ("ppo", "a2c"):
        value_module = ValueOperator(mlp(29, 1, hp["hidden"]), in_keys=["observation"])
        adv_module = GAE(gamma=hp["gamma"], lmbda=hp["lmbda"], value_network=value_module)
        ent_coeff = max(hp["entropy_coeff"], 1e-6)  # 0.0 would hide the entropy output in TorchRL
        if algo == "ppo":
            loss_module = ClipPPOLoss(policy, value_module, clip_epsilon=hp["clip_epsilon"],
                                      entropy_coeff=ent_coeff,
                                      critic_coeff=hp["critic_coeff"],
                                      loss_critic_type="smooth_l1", normalize_advantage=True)
        else:
            loss_module = A2CLoss(policy, value_module, entropy_coeff=ent_coeff,
                                  critic_coeff=hp["critic_coeff"], loss_critic_type="smooth_l1")
        params = list(loss_module.parameters())
    optim = torch.optim.Adam(params, lr=hp["lr"])

    collector = SyncDataCollector(make_env, policy, frames_per_batch=hp["frames_per_batch"],
                                  total_frames=hp["total_frames"], device="cpu")
    try:
        collector.set_seed(seed)
    except Exception:
        pass

    curve_path = os.path.join(log_dir, f"run_{run_id:02d}_curve.csv")
    f = open(curve_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["frames", "mean_episode_return", "loss", "entropy"])

    frames, running, recent = 0, 0.0, []
    t0 = time.time()
    print(f"\n=== {algo.upper()} run {run_id} === {json.dumps(hp)}")

    for i, data in enumerate(collector):
        n = data.numel()
        frames += n
        rewards = data["next", "reward"].reshape(-1)
        dones = data["next", "done"].reshape(-1)

        batch_returns = []
        for r, d in zip(rewards.tolist(), dones.tolist()):
            running += r
            if d:
                batch_returns.append(running)
                running = 0.0
        recent.extend(batch_returns)

        losses, entropies = [], []
        if algo == "reinforce":
            obs = data["observation"].reshape(n, -1)
            act = data["action"].reshape(n)
            G = discounted_returns(rewards.numpy(), dones.numpy(), hp["gamma"])
            if hp["normalize_returns"]:
                G = (G - G.mean()) / (G.std() + 1e-8)
            dist = Categorical(logits=actor_net(obs))
            entropy = dist.entropy().mean()
            loss = -(dist.log_prob(act) * G).mean() - hp["entropy_coeff"] * entropy
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, hp["grad_clip"])
            optim.step()
            losses.append(loss.item())
            entropies.append(entropy.item())

        elif algo == "a2c":
            with torch.no_grad():
                adv_module(data)
            loss_td = loss_module(data)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, hp["grad_clip"])
            optim.step()
            losses.append(loss.item())
            entropies.append(float(loss_td.get("entropy", torch.tensor(float("nan")))))

        else:  # ppo
            with torch.no_grad():
                adv_module(data)
            flat = data.reshape(-1)
            for _ in range(hp["epochs"]):
                perm = torch.randperm(n)
                for s in range(0, n, hp["minibatch"]):
                    sub = flat[perm[s:s + hp["minibatch"]]]
                    loss_td = loss_module(sub)
                    loss = (loss_td["loss_objective"] + loss_td["loss_critic"]
                            + loss_td["loss_entropy"])
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, hp["grad_clip"])
                    optim.step()
                    losses.append(loss.item())
                    entropies.append(float(loss_td.get("entropy", torch.tensor(float("nan")))))

        mean_ret = float(np.mean(batch_returns)) if batch_returns else float("nan")
        mean_loss, mean_ent = float(np.mean(losses)), float(np.mean(entropies))
        writer.writerow([frames, mean_ret, mean_loss, mean_ent])
        if i % max(1, 2000 // hp["frames_per_batch"]) == 0:
            avg = np.mean(recent[-10:]) if recent else float("nan")
            print(f"frames={frames:>6} | avg return (last 10 eps)={avg:>9.1f} | "
                  f"loss={mean_loss:>8.3f} | entropy={mean_ent:.3f}")

    f.close()
    collector.shutdown()

    result = evaluate(actor_net)
    result.update(algo=algo, run=run_id,
                  train_last20=float(np.mean(recent[-20:])) if recent else float("nan"),
                  minutes=round((time.time() - t0) / 60, 2), **hp)
    torch.save({"state_dict": actor_net.state_dict(), "hidden": hp["hidden"]},
               os.path.join(MODEL_DIR, f"{algo}_run_{run_id:02d}.pt"))

    best_path = os.path.join(MODEL_DIR, f"best_{algo}.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else {"eval_mean": -1e9}
    if result["eval_mean"] > best["eval_mean"]:
        torch.save({"state_dict": actor_net.state_dict(), "hidden": hp["hidden"]},
                   os.path.join(MODEL_DIR, f"best_{algo}.pt"))
        json.dump({"run": run_id, "eval_mean": result["eval_mean"]}, open(best_path, "w"))
        print(f"New best {algo.upper()} model: run {run_id}")

    summary_path = os.path.join(log_dir, "summary.csv")
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=list(result.keys()))
        if new_file:
            w.writeheader()
        w.writerow(result)
    print(f"Run {run_id} done: eval mean return = {result['eval_mean']:.1f} "
          f"(+/- {result['eval_std']:.1f}), overdue = {result['eval_overdue']:.1f}, "
          f"time = {result['minutes']} min")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "ppo", "a2c"], required=True)
    p.add_argument("--run", type=int, help="run number 1-12")
    p.add_argument("--all", action="store_true", help="run all 12 configurations")
    p.add_argument("--smoke", action="store_true", help="tiny test run")
    args = p.parse_args()

    runs = get_runs(args.algo)
    if args.smoke:
        train(args.algo, 0, {**runs[0], "total_frames": 3_000})
    elif args.all:
        for i, hp in enumerate(runs, start=1):
            train(args.algo, i, hp)
    elif args.run:
        train(args.algo, args.run, runs[args.run - 1])
    else:
        p.print_help()
