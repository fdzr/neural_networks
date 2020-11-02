import lineal
import functions_activations as FA
from FFNN import FFNN
from FFNN_v2 import FFNN as NN
from dataset import RandomDataSet
from training import loop_FFNN

import torch


if __name__ == "__main__":
    N, F, C = 1000, 300, 10
    # red = FFNN(F, [30, 20], [F.sig, F.relu, F.softmax], 10)
    
    dataset = RandomDataSet(N, F, C)
    # loop_FFNN(dataset, 20, [300, 200], 0.001, 20)
    # red = FFNN(F, [400, 300], [FA.sig, FA.relu, FA.softmax], 10)
    # red.resumen()
    with torch.no_grad():
        loop_FFNN(dataset, 20, [400, 300], 0.01, 50)
    
