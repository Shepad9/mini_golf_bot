"""Run an interactive round.

    python main.py            # random course
    python main.py 42         # fixed seed (reproducible course)

Drag from the ball like a slingshot and release to putt. Pull further for more
power. Sink the ball in the cup in as few strokes as you can.
"""

import sys
from generator import generate_course
from render import InteractiveGame


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if seed is None:
        import random
        seed = random.randint(0, 10_000)
    print(f"Course seed: {seed}")
    course, _ = generate_course(seed=seed)
    InteractiveGame(course).play()


if __name__ == "__main__":
    main()