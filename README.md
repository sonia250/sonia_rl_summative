# Court Motion Triage: Mission-Based Reinforcement Learning

A custom reinforcement-learning environment that models the daily decisions of a registrar clearing the
motion backlog of a High Court (mission: reduce case backlog in Rwanda's courts). Four agents are
trained with **TorchRL** on the same environment and compared:

| Type | Algorithm |
|---|---|
| Value-based | DQN |
| Policy gradient | REINFORCE, PPO, A2C |


## The environment (`environment/custom_env.py`)

- **Start state:** 30 pending motions, 8 working hours per day, day 0.
- **Observation (29 numbers in [0, 1]):** for the 5 oldest motions: type, urgency, hours needed, age,
  documents complete; plus backlog size, hours left today, days left, overdue count.
- **Actions (8, discrete):** hear the motion in slot 0-4, fast-track the oldest procedural motion
  (half an hour, 15% risk of appeal), send back the oldest incomplete motion (fixed after 2 days), end the day.
- **Rewards:** +1.0 / +1.5 / +2.0 for resolving a procedural / interlocutory / substantive motion,
  +2 extra if urgent, +0.2 for sending back an incomplete motion, -0.1 per waiting motion per day,
  -3 when a motion passes its deadline, -0.5 for an invalid action, +10 for clearing the backlog.
- **Terminal conditions:** day 60 reached, or the backlog is empty.
- **Why RL:** hours are limited and new motions arrive at random, so today's choice changes tomorrow's queue
  (quick wins versus urgent long motions versus deadlines).
- **Visualization:** pygame dashboard (`environment/rendering.py`); `environment.export_state()` returns the
  queue as JSON for integration with other systems.

## Run it (uv only)

```
uv sync
uv run main.py            # best PPO agent, GUI + terminal output
```

Options (see `play.py`):

```
uv run play.py --algo dqn          # best DQN agent
uv run play.py --algo a2c --fps 5  # slower animation
uv run play.py --random            # random agent for comparison
```

PPO samples its actions, so each run differs slightly. Press any key in the window to close it at the end.

## Re-train (optional)

```
uv run python training/dqn_training.py --run 1          # DQN, run 1-12 (or --all)
uv run python training/pg_training.py --algo ppo --all  # reinforce | ppo | a2c, 12 runs each
uv run python training/evaluate_pg.py                   # fair evaluation of saved policy-gradient models
uv run python training/plots.py                         # figures and tables in assets/
```

Every algorithm has 12 hyperparameter runs on the same environment and the same 60,000-frame budget.
Logs are in `logs/`, saved models in `models/`, figures in `assets/figures/`, and the full
hyperparameter tables in `assets/report_tables.md`.

## Results (mean return on 20 unseen seeds)

| Algorithm | Best run | Mean of 12 runs | Evaluation |
|---|---|---|---|
| PPO | 223.7 (run 12) | 187.7 | actions sampled from the policy |
| DQN | 201.5 (run 4) | 146.0 | greedy |
| A2C | 188.5 (run 8) | 135.9 | actions sampled from the policy |
| REINFORCE | 151.0 (run 1) | 114.3 | actions sampled from the policy |
| Random agent | -268.9 | | |

Policy-gradient agents learn a probability over actions, so they are evaluated by sampling from it
(greedy scores are also reported in `logs/*/summary_eval.csv`). Scores come from one training seed per
run, so small differences between the top runs are within noise.

## Project structure

```
main.py, play.py          demo of the trained agents
environment/              custom_env.py (Gymnasium environment), rendering.py (pygame)
training/                 dqn_training.py, pg_training.py, evaluate_pg.py, plots.py
models/dqn, models/pg     saved models (best_*.pt = best run per algorithm)
logs/                     training curves and summaries
assets/                   environment screenshot, figures, report tables
tests/                    test_env.py
```
