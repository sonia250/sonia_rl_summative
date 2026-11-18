"""Demo: random agent in the Court Motion Triage environment (GUI)."""
from environment.custom_env import CourtMotionEnv

if __name__ == "__main__":
    env = CourtMotionEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"day={info['day']:>2} backlog={info['backlog']:>2} reward={reward:+.1f}")
        done = terminated or truncated
    env.close()
