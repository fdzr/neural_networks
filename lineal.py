import torch


class Lineal(torch.nn.Module):
    def __init__(self, d0, d1, function_activation):
        '''
            d0: int -> 1-dimension size of the matrix
            d1: int -> 2-dimenison size of the matrix
            function_activation: callable -> activation function
                                            for that layer
        '''

        super(Lineal, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(d0, d1))
        self.bias = torch.nn.Parameter(torch.zeros(d1))
        self.F = function_activation
        self._cache = {}

    def get_parameters(self):
        return self.parameters()

    def resumen(self):
        for name, param in self.named_parameters():
            print(f'{name}:\t{param.size()}')

    def __str__(self):
        return f"W: {self.weight.size()}, b: {self.bias.size()}"

    def forward(self, x):
        '''
            y_pred = F(x @ W + b)
        '''
        u = torch.matmul(x, self.weight) + self.bias
        h = self.F(u)
        self._cache.update({'u':u, 'h': h})
        return h

    def backward(self, x, y, y_pred):
        pass

    def num_parameters(self):
        total = 0
        for p in self.get_parameters():
            total += p.numel()
        return total
