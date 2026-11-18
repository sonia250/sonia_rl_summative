"""Pygame visualization of the Court Motion Triage environment."""
from __future__ import annotations

import os

import numpy as np
import pygame

TYPE_NAMES = ["Procedural", "Interlocutory", "Substantive"]
TYPE_COLORS = [(76, 175, 120), (240, 170, 50), (214, 86, 86)]
BG = (18, 26, 43)
PANEL = (28, 40, 64)
CARD = (38, 54, 86)
CARD_OVERDUE = (88, 40, 48)
WHITE = (235, 240, 250)
GREY = (130, 145, 170)
GOLD = (255, 208, 90)
RED = (225, 80, 80)
GREEN = (95, 205, 135)


class CourtRenderer:
    W, H = 1000, 660

    def __init__(self, mode: str = "human", fps: int = 4):
        self.mode = mode
        self.fps = fps
        if mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Court Motion Triage - RL agent")
        self.clock = pygame.time.Clock()
        self._fonts: dict[int, pygame.font.Font] = {}

    # ---- small helpers ---------------------------------------------------
    def _font(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _text(self, s, size, color, pos):
        self.screen.blit(self._font(size).render(str(s), True, color), pos)

    def _courthouse(self, x, y):
        pygame.draw.polygon(self.screen, GOLD, [(x, y + 16), (x + 30, y), (x + 60, y + 16)])
        for i in range(4):
            pygame.draw.rect(self.screen, WHITE, (x + 6 + i * 14, y + 20, 7, 24))
        pygame.draw.rect(self.screen, GOLD, (x, y + 46, 60, 6))

    # ---- drawing ---------------------------------------------------------
    def draw(self, env):
        s = self.screen
        s.fill(BG)

        # header
        pygame.draw.rect(s, PANEL, (0, 0, self.W, 80))
        self._courthouse(20, 14)
        self._text("Court Motion Triage", 40, WHITE, (100, 14))
        self._text("High Court backlog simulation - agent acts as registrar", 24, GREY, (100, 48))

        # queue cards (2 columns x 5 rows = 10 visible)
        self._text("Motion queue (oldest first)  |  gold border = what the agent sees",
                   22, GREY, (20, 90))
        for i, m in enumerate(env.queue[:10]):
            col, row = i % 2, i // 2
            x, y = 20 + col * 335, 115 + row * 100
            overdue = m["deadline"] < 0
            pygame.draw.rect(s, CARD_OVERDUE if overdue else CARD, (x, y, 325, 90), border_radius=8)
            seen = i < env.VIEW
            pygame.draw.rect(s, GOLD if seen else GREY, (x, y, 325, 90),
                             3 if seen else 1, border_radius=8)
            pygame.draw.rect(s, TYPE_COLORS[m["type"]], (x, y, 8, 90), border_radius=4)
            slot = f"[slot {i}] " if seen else ""
            self._text(f"{slot}#{m['id']}  {TYPE_NAMES[m['type']]}", 26, WHITE, (x + 18, y + 8))
            self._text(f"Needs {m['hours']}h | Deadline {m['deadline']}d | Age {m['age']}d",
                       22, GREY, (x + 18, y + 34))
            if m["complete"]:
                status, color = "READY", GREEN
            elif m["fix_left"] > 0:
                status, color = f"FIXING ({m['fix_left']}d)", GOLD
            else:
                status, color = "MISSING DOCS", RED
            self._text(status, 22, color, (x + 18, y + 60))
            urgency = 1.0 - min(max(m["deadline"], 0), 10) / 10.0
            pygame.draw.rect(s, BG, (x + 150, y + 64, 160, 10), border_radius=4)
            pygame.draw.rect(s, RED if urgency > 0.7 else GOLD,
                             (x + 150, y + 64, int(160 * urgency), 10), border_radius=4)
            self._text("urgency", 18, GREY, (x + 150, y + 50))
        extra = len(env.queue) - 10
        if extra > 0:
            self._text(f"+ {extra} more motions waiting", 24, GOLD, (20, 622))

        # right panel
        px = 700
        pygame.draw.rect(s, PANEL, (px, 90, 280, 560), border_radius=10)
        self._text(f"Day {env.day} / {env.MAX_DAYS}", 34, WHITE, (px + 16, 104))
        self._text("Hours left today", 22, GREY, (px + 16, 146))
        blocks = int(round(env.hours_left * 2))
        for b in range(int(env.HOURS_PER_DAY * 2)):
            c = GREEN if b < blocks else BG
            pygame.draw.rect(s, c, (px + 16 + b * 16, 170, 13, 22), border_radius=3)
        self._text(f"{env.hours_left:.1f} h", 24, WHITE, (px + 16, 198))

        stats = [
            ("Backlog", len(env.queue), WHITE),
            ("Resolved", env.resolved, GREEN),
            ("Overdue (total)", env.overdue_total, RED),
            ("Invalid actions", env.invalid_actions, GOLD),
        ]
        for k, (label, val, color) in enumerate(stats):
            y = 240 + k * 36
            self._text(label, 24, GREY, (px + 16, y))
            self._text(val, 30, color, (px + 200, y - 3))

        self._text("Last action", 22, GREY, (px + 16, 400))
        self._text(env.last_action, 26, WHITE, (px + 16, 424))
        rc = GREEN if env.last_reward > 0 else RED if env.last_reward < 0 else WHITE
        self._text(f"Reward: {env.last_reward:+.1f}", 30, rc, (px + 16, 462))
        self._text("Event", 22, GREY, (px + 16, 508))
        words, line, y = env.last_event.split(" "), "", 530
        for w in words:
            if len(line) + len(w) > 28:
                self._text(line, 22, WHITE, (px + 16, y))
                line, y = "", y + 22
            line += w + " "
        self._text(line, 22, WHITE, (px + 16, y))

        if self.mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def save(self, path: str):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.display.quit()
        pygame.quit()
