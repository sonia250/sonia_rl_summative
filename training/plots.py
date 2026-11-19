"""Make all report figures and tables from the training logs.

Run from the project root (after training and evaluate_pg.py):
    uv run python training/plots.py

Output goes to assets/figures/ (PNG files) and assets/report_tables.md
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOGS = os.path.join(ROOT, "logs")
FIG_DIR = os.path.join(ROOT, "assets", "figures")
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#1f77b4", "reinforce": "#d62728", "ppo": "#2ca02c", "a2c": "#ff7f0e"}
GRID = np.arange(0, 60001, 500)


# ---------------------------------------------------------------- helpers
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_curve(algo, run, col):
    """Curve on a common frame grid (NaNs dropped, then interpolated)."""
    path = os.path.join(LOGS, algo, f"run_{run:02d}_curve.csv")
    if not os.path.exists(path):
        return None
    rows = read_csv(path)
    x = np.array([fnum(r["frames"]) for r in rows])
    y = np.array([fnum(r[col]) for r in rows])
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return None
    return np.interp(GRID, x[ok], y[ok])


def smooth(y, w=7):
    pad = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def load_results():
    """results[algo][run] = dict(score, std, overdue, backlog, params...)."""
    res = {}
    for algo in ALGOS:
        res[algo] = {}
        summary = {int(float(r["run"])): r for r in read_csv(os.path.join(LOGS, algo, "summary.csv"))}
        if algo == "dqn":
            for run, r in summary.items():
                res[algo][run] = dict(score=fnum(r["eval_mean"]), std=fnum(r["eval_std"]),
                                      overdue=fnum(r["eval_overdue"]), backlog=fnum(r["eval_backlog"]),
                                      train=fnum(r["train_last20"]), params=r)
        else:
            ev = {int(float(r["run"])): r for r in read_csv(os.path.join(LOGS, algo, "summary_eval.csv"))}
            for run, r in summary.items():
                if run in ev:
                    e = ev[run]
                    res[algo][run] = dict(score=fnum(e["sampled_mean"]), std=fnum(e["sampled_std"]),
                                          overdue=fnum(e["sampled_overdue"]),
                                          backlog=fnum(e["sampled_backlog"]),
                                          greedy=fnum(e["greedy_mean"]),
                                          train=fnum(r["train_last20"]), params=r)
    return res


def best_run(res, algo):
    return max(res[algo], key=lambda r: res[algo][r]["score"])


def random_baseline(n=20, seed0=10_000):
    from environment.custom_env import CourtMotionEnv
    env = CourtMotionEnv()
    rets, od, bl = [], [], []
    for ep in range(n):
        env.reset(seed=seed0 + ep)
        env.action_space.seed(seed0 + ep)
        done, tot = False, 0.0
        while not done:
            _, r, t, tr, info = env.step(env.action_space.sample())
            tot += r
            done = t or tr
        rets.append(tot)
        od.append(info["overdue_total"])
        bl.append(info["backlog"])
    return dict(score=float(np.mean(rets)), std=float(np.std(rets)),
                overdue=float(np.mean(od)), backlog=float(np.mean(bl)))


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=160)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- figures
def fig_reward_comparison(res):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for algo in ALGOS:
        runs = list(res[algo])
        curves = [load_curve(algo, r, "mean_episode_return") for r in runs]
        curves = [smooth(c) for c in curves if c is not None]
        best = load_curve(algo, best_run(res, algo), "mean_episode_return")
        ax.plot(GRID, smooth(best), color=COLORS[algo], lw=2.2,
                label=f"{NAMES[algo]} - best run ({best_run(res, algo)})")
        ax.plot(GRID, np.mean(curves, axis=0), color=COLORS[algo], lw=1.2, ls="--",
                label=f"{NAMES[algo]} - mean of {len(curves)} runs")
    ax.set_xlabel("Environment steps (frames)")
    ax.set_ylabel("Episode return (training)")
    ax.set_title("Cumulative reward during training: all four algorithms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig1_reward_comparison.png")


def fig_dqn_loss(res):
    best = best_run(res, "dqn")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in res["dqn"]:
        c = load_curve("dqn", r, "loss")
        if c is not None:
            axes[0].plot(GRID, smooth(c), color="grey", alpha=0.35, lw=0.9)
    axes[0].plot(GRID, smooth(load_curve("dqn", best, "loss")), color=COLORS["dqn"], lw=2.2,
                 label=f"best run ({best})")
    axes[0].set(xlabel="Frames", ylabel="DQN loss (smooth L1)", title="DQN objective (TD loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(GRID, smooth(load_curve("dqn", best, "mean_q")), color="purple", lw=2)
    axes[1].set(xlabel="Frames", ylabel="Mean max Q-value", title=f"DQN Q-values (run {best})")
    axes[1].grid(alpha=0.3)
    save(fig, "fig2_dqn_loss.png")


def fig_pg_entropy(res):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, algo in zip(axes, ["reinforce", "ppo", "a2c"]):
        for r in res[algo]:
            c = load_curve(algo, r, "entropy")
            if c is not None:
                ax.plot(GRID, smooth(c), color="grey", alpha=0.35, lw=0.9)
        b = best_run(res, algo)
        ax.plot(GRID, smooth(load_curve(algo, b, "entropy")), color=COLORS[algo], lw=2.2,
                label=f"best run ({b})")
        ax.axhline(np.log(8), color="black", ls=":", lw=1, label="random policy (ln 8)")
        ax.set(title=NAMES[algo], xlabel="Frames")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Policy entropy")
    fig.suptitle("Policy entropy (exploration) during training", y=1.02)
    save(fig, "fig3_policy_entropy.png")


def convergence_frames(curve, threshold=100.0):
    idx = np.where(smooth(curve) >= threshold)[0]
    return int(GRID[idx[0]]) if len(idx) else None


def fig_convergence(res):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for r in res[algo]:
            c = load_curve(algo, r, "mean_episode_return")
            if c is not None:
                ax.plot(GRID, smooth(c), color="grey", alpha=0.4, lw=0.9)
        b = best_run(res, algo)
        ax.plot(GRID, smooth(load_curve(algo, b, "mean_episode_return")), color=COLORS[algo], lw=2.2)
        ax.axhline(100, color="black", ls=":", lw=1)
        ax.set(title=f"{NAMES[algo]} (best = run {b})", xlabel="Frames")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Episode return")
    fig.suptitle("Convergence: all 12 runs per algorithm (grey) and best run (colour); dotted = return 100", y=1.03)
    save(fig, "fig4_convergence.png")


def fig_generalization(res, base):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, data = [], []
    for algo in ALGOS:
        labels.append(f"{NAMES[algo]}\n({'greedy' if algo == 'dqn' else 'sampled'})")
        data.append([v["score"] for v in res[algo].values()])
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, algo in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.35)
    for i, algo in enumerate(ALGOS, start=1):
        b = best_run(res, algo)
        v = res[algo][b]
        ax.errorbar(i, v["score"], yerr=v["std"], fmt="D", color="black", capsize=4,
                    label="best run (+/- std over 20 seeds)" if i == 1 else None)
        ax.scatter([i] * len(data[i - 1]), data[i - 1], color=COLORS[algo], s=14, zorder=3)
    ax.axhline(base["score"], color="red", ls="--", label=f"random agent ({base['score']:.0f})")
    ax.set_ylabel("Mean return on 20 unseen seeds")
    ax.set_title("Evaluation on unseen episodes (12 runs per algorithm)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "fig5_generalization.png")


def fig_behaviour(res, base):
    names = ["Random"] + [NAMES[a] for a in ALGOS]
    od = [base["overdue"]] + [res[a][best_run(res, a)]["overdue"] for a in ALGOS]
    bl = [base["backlog"]] + [res[a][best_run(res, a)]["backlog"] for a in ALGOS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, od, 0.4, label="Overdue motions (episode total)", color="#d62728")
    ax.bar(x + 0.2, bl, 0.4, label="Backlog at end of term", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Agent behaviour of the best run per algorithm")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "fig6_behaviour.png")


# ---------------------------------------------------------------- tables
HP_COLS = {
    "dqn": ["lr", "gamma", "batch_size", "buffer_size", "hidden", "eps_end", "eps_anneal", "tau", "utd"],
    "reinforce": ["lr", "gamma", "frames_per_batch", "entropy_coeff", "normalize_returns", "hidden"],
    "ppo": ["lr", "gamma", "lmbda", "frames_per_batch", "clip_epsilon", "entropy_coeff", "epochs",
            "minibatch", "hidden"],
    "a2c": ["lr", "gamma", "lmbda", "frames_per_batch", "entropy_coeff", "critic_coeff", "hidden"],
}


def fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) and abs(f) >= 10 else f"{f:g}"
    except ValueError:
        return str(v)


def write_tables(res, base):
    lines = ["# Report tables (generated)\n"]
    lines.append(f"Random agent baseline: return {base['score']:.1f} (+/- {base['std']:.1f}), "
                 f"overdue {base['overdue']:.1f}, backlog {base['backlog']:.1f}\n")
    for algo in ALGOS:
        cols = HP_COLS[algo]
        how = "greedy" if algo == "dqn" else "sampled from the policy"
        lines.append(f"\n## {NAMES[algo]} - 12 hyperparameter runs (evaluation: {how}, 20 unseen seeds)\n")
        lines.append("| Run | " + " | ".join(cols) + " | Train return (last 20 eps) | Eval return | Overdue | Backlog |")
        lines.append("|" + "---|" * (len(cols) + 5))
        for run in sorted(res[algo]):
            v = res[algo][run]
            hp = " | ".join(fmt(v["params"][c]) for c in cols)
            lines.append(f"| {run} | {hp} | {v['train']:.1f} | {v['score']:.1f} +/- {v['std']:.1f} | "
                         f"{v['overdue']:.1f} | {v['backlog']:.1f} |")
    lines.append("\n## Summary across algorithms\n")
    lines.append("| Algorithm | Best run | Best eval | Mean of 12 runs | Worst run | Steps to reach return 100 (best run) |")
    lines.append("|---|---|---|---|---|---|")
    for algo in ALGOS:
        b = best_run(res, algo)
        scores = [v["score"] for v in res[algo].values()]
        cf = convergence_frames(load_curve(algo, b, "mean_episode_return"))
        lines.append(f"| {NAMES[algo]} | {b} | {res[algo][b]['score']:.1f} | {np.mean(scores):.1f} | "
                     f"{min(scores):.1f} | {cf if cf is not None else 'not reached'} |")
    with open(os.path.join(ROOT, "assets", "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved assets/report_tables.md")


if __name__ == "__main__":
    results = load_results()
    baseline = random_baseline()
    fig_reward_comparison(results)
    fig_dqn_loss(results)
    fig_pg_entropy(results)
    fig_convergence(results)
    fig_generalization(results, baseline)
    fig_behaviour(results, baseline)
    write_tables(results, baseline)
    print("\nBest run per algorithm:")
    for a in ALGOS:
        b = best_run(results, a)
        print(f"  {NAMES[a]:<10} run {b:>2}  eval = {results[a][b]['score']:.1f}")
    print(f"  Random     eval = {baseline['score']:.1f}")
