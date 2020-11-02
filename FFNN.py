from lineal import Lineal

import torch


class FFNN(torch.nn.Module):
    def __init__(self, F, l_h, l_a, C):
        super(FFNN, self).__init__()

        dim = [F] + l_h + [C]
        self.layers = [Lineal(dim[i], dim[i+1], l_a[i])for i in range(len(dim) - 1)]
        self.functions_activation = l_a
        self.parameters_tmp = []
        for layer in self.layers:
            self.parameters_tmp.append(layer.weight)
            self.parameters_tmp.append(layer.bias)
        self.parameters = torch.nn.ParameterList(self.parameters_tmp)
        # for i in self.parameters():
        #     print(i.size())

    def resumen(self):
        for name, parameter in self.named_parameters():
            print(f'{name}\t{parameter.size()}')

    def get_layers(self):
        for layer in self.layers:
            print(layer._cache['u'].size())

    def num_parameters(self):
        total = 0
        for layer in self.layers:
            total += layer.num_parameters()
        return total

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, y_pred):
        B = x.size()[0]
        dL_du = (1/B) * (y_pred - y)

        grads = []

        layers = self.layers[::-1]

        for index in range(len(layers) - 1):
            dL_dW = layers[index + 1]._cache['h'].t() @ dL_du
            dL_db = torch.sum(dL_du, 0)
            dL_dh = dL_du @ layers[index].weight.t()
            dL_du = dL_dh * layers[index + 1].F(layers[index + 1]._cache['u'], gradient=True)
            grads.insert(0, dL_db)
            grads.insert(0, dL_dW)

        dL_dW = x.t() @ dL_du
        dL_db = torch.sum(dL_du, 0)
        grads.insert(0, dL_db)
        grads.insert(0, dL_dW)

        for p, g in zip(self.parameters_tmp, grads):
            p.grad = g
