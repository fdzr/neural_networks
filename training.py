import time
import sys

import torch
from torch.utils.data import DataLoader

from FFNN import FFNN
from FFNN_v2 import FFNN as NN
from loss import CELoss
from optimizer import SGD
import functions_activations as F_A


def loop_FFNN(dataset, batch_size, l_h, lr, 
                 epochs, run_in_GPU=False, 
                 cheq_grad=False, reports_every=1):

    device = 'cuda' if run_in_GPU else 'cpu'

    F = dataset.num_features

    # red = FFNN(F, l_h, [F_A.sig, F_A.relu, F_A.softmax], dataset.Y.size()[1])
    red = NN(F, l_h, [F_A.sig, F_A.relu], dataset.Y.size()[1])

    red.to(device)

    print('Cantidad de parámetros:', red.num_parameters())

    data = DataLoader(dataset, batch_size, shuffle=True)

    optimizer = SGD(red.parameters(), lr)

    tiempo_epochs = 0

    for e in range(1, epochs + 1):
        inicio_epoch = time.clock()

        for x, y in data:
            x,y = x.to(device), y.to(device)
            
            y_pred = red.forward(x)

            L = CELoss(y_pred, y)

            red.backward(x, y, y_pred)

            optimizer.step()

        tiempo_epochs += time.clock() - inicio_epoch

        if e % reports_every == 0:
            X = dataset.X.to(device)
            Y = dataset.Y.to(device)

            # Predice usando la red
            Y_PRED = red.forward(X)

            L_total = CELoss(Y_PRED, Y)

            Y_PRED_MC = torch.max(Y_PRED, 1)[1]

            correctos = torch.sum(Y_PRED_MC == torch.max(Y, 1)[1])

            acc = (correctos.item() / X.size()[0]) * 100

            sys.stdout.write(
            '\rEpoch:{0:03d}'.format(e) + ' Acc:{0:.2f}%'.format(acc)
            + ' Loss:{0:.4f}'.format(L_total) 
            + ' Tiempo/epoch:{0:.3f}s'.format(tiempo_epochs/e))

    print()