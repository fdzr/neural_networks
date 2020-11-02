import torch

from functions_activations import softmax


class FFNN(torch.nn.Module):
    def __init__(self, F, l_h, l_a, C):
        super(FFNN, self).__init__()

        dim = [F] + l_h + [C]
        self.Ws = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(dim[i], dim[i + 1])) 
                    for i in range(len(dim) - 1)]
        )
        self.bs = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(i))
                    for i in dim[1:]]
        )
        self.f_a = l_a
        self._cache = {}

    def load_weights(self, Ws, U, bs, c):
        self.Ws = torch.nn.ParameterList([torch.nn.Parameter(W) for W in Ws + [U]])
        self.bs = torch.nn.ParameterList([torch.nn.Parameter(b) for b in bs + [c]])

    def parameters(self):
        return self.concatenate(self.Ws, self.bs)

    def resumen(self):
        for name, parameter in self.named_parameters():
            print(f'{name}\t{parameter.size()}')

    def concatenate(self, l1, l2):
        tmp = []
        for i in zip(l1, l2):
            tmp.append(i[0])
            tmp.append(i[1])
        return tmp

    def num_parameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total

    def forward(self, x):
        index = 1
        for W, b, F in zip(self.Ws, self.bs, self.f_a):
            u = torch.matmul(x, W) + b
            x = F(u)
            self._cache.update({f'u{index}': u, f'h{index}': x})
            index += 1
        return softmax(torch.matmul(x, self.Ws[-1]) + self.bs[-1])

    def backward(self, x, y, y_pred):
        B = x.size()[0]
        dL_du = (1/B) * (y_pred - y)

        grads = []
        f_a = self.f_a[::-1]
        Ws = self.Ws[::-1]

        index = int(len(self._cache) / 2)

        for F, W in zip(f_a, Ws):
            dL_dW = self._cache[f'h{index}'].t() @ dL_du
            dL_db = torch.sum(dL_du, 0)
            dL_dh = dL_du @ W.t()
            dL_du = dL_dh * F(self._cache[f'u{index}'], gradient=True)
            index -= 1
            grads.insert(0, dL_db)
            grads.insert(0, dL_dW)

        dL_dW = x.t() @ dL_du
        dL_db = torch.sum(dL_du, 0)
        grads.insert(0, dL_db)
        grads.insert(0, dL_dW)

        params = self.concatenate(self.Ws, self.bs)

        for p, g in zip(params, grads):
            p.grad = g
