import sys
from enum import Enum
import numpy as np
import seaborn as sns

class slot(Enum):
    AGENT: 10
    OBSTACLE: -1
    GROUND_INITIAL: 0
    GROUND_GOAL: 7
    GROUND_DEFEAT: 4

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 TP2.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    f = open(input_file_name, "r")

    params = f.readline().split()
    epsilon = -1
    if(len(params) > 4): 
        nIters, learningRate, discountFactor, stdReward, epsilon = map(float, params)
    else:
        nIters, learningRate, discountFactor, stdReward = map(float, params)
    nIters = int(nIters)
    
    N = int(f.readline())

    Grid = np.zeros((N,N))
    for i in range(N): Grid[i] = list(map(int, f.readline().split()))
    print(Grid)
    print(nIters, learningRate, discountFactor, stdReward, epsilon)
    f.close()

    sns.heatmap(Grid, square=True, cbar=False)