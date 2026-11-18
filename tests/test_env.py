"""Run from project root:  uv run python tests/test_env.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium.utils.env_checker import check_env

from environment.custom_env import CourtMotionEnv


def test_gymnasium_api():
    check_env(CourtMotionEnv(), skip_render_check=True)


def test_random_episode():
    env = CourtMotionEnv()
    obs, info = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        done = terminated or truncated
    print(f"Random agent: reward={total:.1f}, resolved={info['resolved']}, "
          f"overdue={info['overdue_total']}, backlog={info['backlog']}, day={info['day']}")


def save_screenshot():
    env = CourtMotionEnv(render_mode="rgb_array")
    env.reset(seed=1)
    for _ in range(25):
        env.step(env.action_space.sample())
    frame = env.render()
    env._renderer.save(os.path.join("assets", "environment_screenshot.png"))
    print("Frame shape:", frame.shape, "- saved assets/environment_screenshot.png")
    env.close()


if __name__ == "__main__":
    test_gymnasium_api()
    print("check_env passed")
    test_random_episode()
    save_screenshot()
