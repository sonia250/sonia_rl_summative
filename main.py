"""Entry point: `uv sync` then `uv run main.py`.

Runs the best trained agent (PPO) in the Court Motion Triage environment with the
GUI and terminal output. See play.py for options (other algorithms, speed, seeds).
"""
from play import main

if __name__ == "__main__":
    main()
