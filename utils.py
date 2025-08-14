import numpy as np

def judge_convergence(data, name, window, threshold):
    if len(data) < window:
        print(f"Not enough data for convergence check: {name}")
        return False
    std = np.std(data[-window:])
    print(f"Convergence check for {name}: std={std:.5f}")
    return std < threshold
