"""Court Motion Triage - a custom Gymnasium environment.

Mission: reduce the motion backlog at a High Court in Rwanda.
Each step the agent (a registrar/judge) decides what to do with a limited
number of working hours. Choices made today change the queue tomorrow,
so the problem is sequential and needs RL.
"""
from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TYPE_NAMES = ["Procedural", "Interlocutory", "Substantive"]
TYPE_PROBS = [0.5, 0.3, 0.2]          # how often each type arrives
TYPE_HOURS = [(1, 1), (2, 2), (4, 5)]  # hours needed (min, max) per type

ACTION_NAMES = [
    "Hear slot 0",
    "Hear slot 1",
    "Hear slot 2",
    "Hear slot 3",
    "Hear slot 4",
    "Fast-track oldest procedural",
    "Send back oldest incomplete",
    "End the day",
]


class CourtMotionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ---- environment settings -------------------------------------------
    VIEW = 5               # motions the agent can see (oldest first)
    FEATURES = 5           # numbers describing each visible motion
    HOURS_PER_DAY = 8.0
    MAX_DAYS = 60          # terminal condition 1
    INITIAL_MOTIONS = 30   # start state
    ARRIVAL_RATE = 3.0     # average new motions per day
    MAX_BACKLOG = 60
    MAX_STEPS = 1500       # safety limit (truncation)
    URGENT_DAYS = 3        # deadline <= 3 days counts as urgent
    FIX_DAYS = 2           # days to correct a sent-back motion
    APPEAL_PROB = 0.15     # risk when fast-tracking

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.VIEW * self.FEATURES + 4,), dtype=np.float32,
        )
        self._renderer = None
        self.queue: list[dict] = []

    # ---- helpers ---------------------------------------------------------
    def _new_motion(self, initial: bool = False) -> dict:
        t = int(self.np_random.choice(3, p=TYPE_PROBS))
        lo, hi = TYPE_HOURS[t]
        m = {
            "id": self._next_id,
            "type": t,
            "hours": int(self.np_random.integers(lo, hi + 1)),
            "deadline": int(self.np_random.integers(2, 16)),  # days left
            "age": int(self.np_random.integers(0, 8)) if initial else 0,
            "complete": bool(self.np_random.random() > 0.25),  # 25% missing docs
            "fix_left": 0,
            "penalized": False,
        }
        self._next_id += 1
        return m

    def _sort_queue(self):
        self.queue.sort(key=lambda m: (-m["age"], m["id"]))  # oldest first

    def _overdue_count(self) -> int:
        return sum(1 for m in self.queue if m["deadline"] < 0)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.VIEW * self.FEATURES + 4, dtype=np.float32)
        for i, m in enumerate(self.queue[: self.VIEW]):
            urgency = 1.0 - min(max(m["deadline"], 0), 10) / 10.0
            obs[i * 5:(i + 1) * 5] = [
                m["type"] / 2.0,
                urgency,
                m["hours"] / 5.0,
                min(m["age"], 30) / 30.0,
                1.0 if m["complete"] else 0.0,
            ]
        b = self.VIEW * self.FEATURES
        obs[b] = len(self.queue) / self.MAX_BACKLOG
        obs[b + 1] = max(self.hours_left, 0.0) / self.HOURS_PER_DAY
        obs[b + 2] = (self.MAX_DAYS - self.day) / self.MAX_DAYS
        obs[b + 3] = min(self._overdue_count() / self.MAX_BACKLOG, 1.0)
        return np.clip(obs, 0.0, 1.0)

    def _info(self) -> dict:
        return {
            "day": self.day,
            "backlog": len(self.queue),
            "hours_left": self.hours_left,
            "resolved": self.resolved,
            "overdue_total": self.overdue_total,
            "invalid_actions": self.invalid_actions,
        }

    def export_state(self) -> str:
        """JSON snapshot of the environment (used for the API/serialization path)."""
        return json.dumps({
            "day": self.day,
            "hours_left": self.hours_left,
            "queue": self.queue,
            "resolved": self.resolved,
            "overdue_total": self.overdue_total,
        })

    # ---- actions ---------------------------------------------------------
    def _resolve(self, m: dict, base: float, msg: str):
        bonus = 2.0 if m["deadline"] <= self.URGENT_DAYS else 0.0
        self.queue.remove(m)
        self.resolved += 1
        return base + bonus, msg + (" (urgent +2)" if bonus else "")

    def _hear(self, slot: int):
        if slot >= len(self.queue):
            return None
        m = self.queue[slot]
        if not m["complete"] or m["hours"] > self.hours_left:
            return None
        self.hours_left -= m["hours"]
        return self._resolve(m, 1.0 + 0.5 * m["type"], f"Heard motion #{m['id']}")

    def _fast_track(self):
        cands = [m for m in self.queue if m["type"] == 0 and m["complete"]]
        if not cands or self.hours_left < 0.5:
            return None
        m = cands[0]
        self.hours_left -= 0.5
        r, msg = self._resolve(m, 1.0, f"Fast-tracked #{m['id']}")
        if self.np_random.random() < self.APPEAL_PROB:
            r -= 1.0
            msg += " - appealed (-1)"
        return r, msg

    def _send_back(self):
        cands = [m for m in self.queue if not m["complete"] and m["fix_left"] == 0]
        if not cands or self.hours_left < 0.5:
            return None
        m = cands[0]
        m["fix_left"] = self.FIX_DAYS
        self.hours_left -= 0.5
        return 0.2, f"Sent back #{m['id']} for correction"

    def _end_day(self) -> float:
        r = 0.0
        self.day += 1
        for m in self.queue:
            m["age"] += 1
            m["deadline"] -= 1
            if m["fix_left"] > 0:
                m["fix_left"] -= 1
                if m["fix_left"] == 0:
                    m["complete"] = True
            if m["deadline"] < 0 and not m["penalized"]:
                m["penalized"] = True
                self.overdue_total += 1
                r -= 3.0
        r -= 0.1 * len(self.queue)  # waiting cost
        n_new = int(self.np_random.poisson(self.ARRIVAL_RATE))
        n_new = min(n_new, self.MAX_BACKLOG - len(self.queue))
        for _ in range(n_new):
            self.queue.append(self._new_motion())
        self.hours_left = self.HOURS_PER_DAY
        self._sort_queue()
        return r

    # ---- Gymnasium API ---------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.hours_left = self.HOURS_PER_DAY
        self.steps = 0
        self.resolved = 0
        self.overdue_total = 0
        self.invalid_actions = 0
        self._next_id = 1
        self.last_action = "-"
        self.last_reward = 0.0
        self.last_event = "New court term begins"
        self.queue = [self._new_motion(initial=True) for _ in range(self.INITIAL_MOTIONS)]
        self._sort_queue()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def step(self, action):
        action = int(action)
        self.steps += 1

        if action < self.VIEW:
            res = self._hear(action)
        elif action == 5:
            res = self._fast_track()
        elif action == 6:
            res = self._send_back()
        else:
            res = (self._end_day(), "Day ended")

        if res is None:  # invalid action: small penalty, costs half an hour
            reward, event = -0.5, "Invalid action (-0.5)"
            self.hours_left = max(self.hours_left - 0.5, 0.0)
            self.invalid_actions += 1
        else:
            reward, event = res

        terminated = False
        if not self.queue:
            reward += 10.0
            terminated = True
            event += " | Backlog cleared! (+10)"
        else:
            if action != 7 and self.hours_left < 0.5:
                reward += self._end_day()
                event += " | Day ended"
            if self.day >= self.MAX_DAYS:
                terminated = True
                event += " | Term over"

        truncated = self.steps >= self.MAX_STEPS
        self.last_action = ACTION_NAMES[action]
        self.last_reward = reward
        self.last_event = event

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import CourtRenderer
            self._renderer = CourtRenderer(self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
