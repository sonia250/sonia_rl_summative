"""DQN (value-based) training with TorchRL on the Court Motion Triage environment.

Run from the project root:
    uv run python training/dqn_training.py --smoke        # 1-minute test
    uv run python training/dqn_training.py --run 1        # one hyperparameter run
    uv run python training/dqn_training.py --all         # all 12 runs
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tensordict.nn import TensorDictSequential
try:  # TorchRL >= 0.11 renamed the collector
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from environment.custom_env import CourtMotionEnv

LOG_DIR = os.path.join(ROOT, "logs", "dqn")
MODEL_DIR = os.path.join(ROOT, "models", "dqn")

# ---------------------------------------------------------------------------
# 12 hyperparameter combinations (run 1 = baseline). Each run changes
# different settings so the report table shows real variation.
# ---------------------------------------------------------------------------
BASE = dict(
    lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50_000, hidden=128,
    eps_end=0.05, eps_anneal=20_000, tau=0.005, utd=0.25,   # utd = updates per frame
    total_frames=60_000, frames_per_batch=200, warmup=1_000,
)
CHANGES = [
    {},                                                    # 1 baseline
    {"lr": 5e-4},                                          # 2 lower learning rate
    {"lr": 1e-4},                                          # 3 much lower learning rate
    {"gamma": 0.95},                                       # 4 short-sighted agent
    {"gamma": 0.999},                                      # 5 very long-sighted agent
    {"batch_size": 128},                                   # 6 bigger batch
    {"buffer_size": 10_000},                               # 7 small replay buffer
    {"eps_anneal": 40_000},                                # 8 slower exploration decay
    {"eps_end": 0.20, "eps_anneal": 10_000},               # 9 more exploration, faster decay
    {"tau": 0.05},                                         # 10 faster target network update
    {"hidden": 256, "lr": 5e-4},                           # 11 bigger network
    {"tau": 0.001, "utd": 0.5, "gamma": 0.98},             # 12 slow target, more updates
]
RUNS = [{**BASE, **c} for c in CHANGES]


def make_env():
    return GymWrapper(CourtMotionEnv(), categorical_action_encoding=True)


def build_net(hidden):
    return nn.Sequential(
        nn.Linear(29, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 8),
    )


def evaluate(net, n_episodes=20, seed0=10_000):
    """Greedy policy on NEW random seeds (generalization check)."""
    rets, overdue, backlog, resolved = [], [], [], []
    env = CourtMotionEnv()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                q = net(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, term, trunc, info = env.step(int(q.argmax()))
            total += r
            done = term or trunc
        rets.append(total)
        overdue.append(info["overdue_total"])
        backlog.append(info["backlog"])
        resolved.append(info["resolved"])
    return dict(eval_mean=float(np.mean(rets)), eval_std=float(np.std(rets)),
                eval_overdue=float(np.mean(overdue)), eval_backlog=float(np.mean(backlog)),
                eval_resolved=float(np.mean(resolved)))


def train(run_id, hp):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env()
    net = build_net(hp["hidden"])
    qvalue_actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(spec=env.action_spec, eps_init=1.0, eps_end=hp["eps_end"],
                            annealing_num_steps=hp["eps_anneal"])
    policy = TensorDictSequential(qvalue_actor, egreedy)

    collector = SyncDataCollector(make_env, policy, frames_per_batch=hp["frames_per_batch"],
                                  total_frames=hp["total_frames"], device="cpu")
    try:
        collector.set_seed(seed)
    except Exception:
        pass

    rb = ReplayBuffer(storage=LazyTensorStorage(hp["buffer_size"]), batch_size=hp["batch_size"])
    loss_module = DQNLoss(qvalue_actor, action_space="categorical", delay_value=True,
                          loss_function="smooth_l1")
    loss_module.make_value_estimator(gamma=hp["gamma"])
    try:
        target_updater = SoftUpdate(loss_module, tau=hp["tau"])
    except TypeError:
        target_updater = SoftUpdate(loss_module, eps=1.0 - hp["tau"])
    optim = torch.optim.Adam(loss_module.parameters(), lr=hp["lr"])

    curve_path = os.path.join(LOG_DIR, f"run_{run_id:02d}_curve.csv")
    f = open(curve_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["frames", "mean_episode_return", "loss", "mean_q", "epsilon"])

    frames, running, recent = 0, 0.0, []
    t0 = time.time()
    print(f"\n=== DQN run {run_id} === {json.dumps({k: v for k, v in hp.items()})}")

    for i, data in enumerate(collector):
        n = data.numel()
        frames += n
        rb.extend(data.reshape(-1))
        egreedy.step(n)

        # episode returns finished in this batch
        rewards = data["next", "reward"].reshape(-1).tolist()
        dones = data["next", "done"].reshape(-1).tolist()
        batch_returns = []
        for r, d in zip(rewards, dones):
            running += r
            if d:
                batch_returns.append(running)
                running = 0.0
        recent.extend(batch_returns)

        loss_val, mean_q = float("nan"), float("nan")
        if len(rb) >= hp["warmup"]:
            losses = []
            for _ in range(max(1, int(n * hp["utd"]))):
                sample = rb.sample()
                loss_td = loss_module(sample)
                loss = loss_td["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
                target_updater.step()
                losses.append(loss.item())
            loss_val = float(np.mean(losses))
            with torch.no_grad():
                mean_q = net(sample["observation"]).max(-1).values.mean().item()

        mean_ret = float(np.mean(batch_returns)) if batch_returns else float("nan")
        writer.writerow([frames, mean_ret, loss_val, mean_q, float(egreedy.eps)])
        if i % 10 == 0:
            avg = np.mean(recent[-10:]) if recent else float("nan")
            print(f"frames={frames:>6} | avg return (last 10 eps)={avg:>9.1f} | "
                  f"loss={loss_val:>8.3f} | eps={float(egreedy.eps):.2f}")

    f.close()
    collector.shutdown()

    result = evaluate(net)
    result.update(run=run_id, train_last20=float(np.mean(recent[-20:])) if recent else float("nan"),
                  minutes=round((time.time() - t0) / 60, 2), **hp)
    torch.save(net.state_dict(), os.path.join(MODEL_DIR, f"dqn_run_{run_id:02d}.pt"))

    # keep the best model (by evaluation score)
    best_path = os.path.join(MODEL_DIR, "best.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else {"eval_mean": -1e9}
    if result["eval_mean"] > best["eval_mean"]:
        torch.save({"state_dict": net.state_dict(), "hidden": hp["hidden"]},
                   os.path.join(MODEL_DIR, "best_dqn.pt"))
        json.dump({"run": run_id, "eval_mean": result["eval_mean"]}, open(best_path, "w"))
        print(f"New best DQN model: run {run_id}")

    summary_path = os.path.join(LOG_DIR, "summary.csv")
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
    p.add_argument("--run", type=int, help="run number 1-12")
    p.add_argument("--all", action="store_true", help="run all 12 configurations")
    p.add_argument("--smoke", action="store_true", help="tiny test run")
    args = p.parse_args()

    if args.smoke:
        hp = {**BASE, "total_frames": 3_000, "warmup": 500, "eps_anneal": 2_000}
        train(0, hp)
    elif args.all:
        for i, hp in enumerate(RUNS, start=1):
            train(i, hp)
    elif args.run:
        train(args.run, RUNS[args.run - 1])
    else:
        p.print_help()
