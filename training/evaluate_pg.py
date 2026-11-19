"""Re-evaluate all saved REINFORCE / PPO / A2C models fairly.

Policy-gradient agents learn a PROBABILITY over actions, so the fair test is to
sample actions from that policy (stochastic). We also report the greedy
(argmax) score for comparison. Same 20 unseen seeds for every model.

Run from the project root AFTER the training runs finish:
    uv run python training/evaluate_pg.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.custom_env import CourtMotionEnv

MODEL_DIR = os.path.join(ROOT, "models", "pg")


def mlp(n_in, n_out, hidden):  # same architecture as in pg_training.py
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def run_episodes(net, stochastic, n_episodes=20, seed0=10_000):
    rets, overdue, backlog, resolved = [], [], [], []
    env = CourtMotionEnv()
    for ep in range(n_episodes):
        torch.manual_seed(seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                logits = net(torch.as_tensor(obs, dtype=torch.float32))
            action = int(Categorical(logits=logits).sample()) if stochastic else int(logits.argmax())
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        rets.append(total)
        overdue.append(info["overdue_total"])
        backlog.append(info["backlog"])
        resolved.append(info["resolved"])
    return (float(np.mean(rets)), float(np.std(rets)), float(np.mean(overdue)),
            float(np.mean(backlog)), float(np.mean(resolved)))


def main():
    for algo in ["reinforce", "ppo", "a2c"]:
        files = sorted(glob.glob(os.path.join(MODEL_DIR, f"{algo}_run_*.pt")))
        rows, best = [], None
        for path in files:
            run = int(re.search(r"run_(\d+)", path).group(1))
            if run == 0:
                continue
            ckpt = torch.load(path, map_location="cpu")
            net = mlp(29, 8, ckpt["hidden"])
            net.load_state_dict(ckpt["state_dict"])
            g = run_episodes(net, stochastic=False)
            s = run_episodes(net, stochastic=True)
            row = dict(algo=algo, run=run, greedy_mean=g[0], greedy_std=g[1],
                       sampled_mean=s[0], sampled_std=s[1], sampled_overdue=s[2],
                       sampled_backlog=s[3], sampled_resolved=s[4])
            rows.append(row)
            print(f"{algo} run {run:>2}: sampled = {s[0]:8.1f} (+/- {s[1]:.1f}) | greedy = {g[0]:8.1f}")
            if best is None or s[0] > best["sampled_mean"]:
                best = row
                best_ckpt = ckpt
        if not rows:
            print(f"No models found for {algo}")
            continue
        out = os.path.join(ROOT, "logs", algo, "summary_eval.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        torch.save(best_ckpt, os.path.join(MODEL_DIR, f"best_{algo}.pt"))
        json.dump({"run": best["run"], "eval_mean": best["sampled_mean"]},
                  open(os.path.join(MODEL_DIR, f"best_{algo}.json"), "w"))
        print(f"==> best {algo}: run {best['run']} (sampled mean {best['sampled_mean']:.1f}); "
              f"saved logs/{algo}/summary_eval.csv\n")


if __name__ == "__main__":
    main()
