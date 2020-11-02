import torch


def softmax(T, gradient=False, dim=1, estable=True):
    if estable:
        T -= T.max(dim=dim, keepdim=True)[0]  
        exp = torch.exp(T)
    if gradient:
        u = softmax(T)
        return u * (1 - u)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def tanh(T, gradient=False):
    E = torch.exp(T)
    e = torch.exp(-1 * T)
    if gradient:
        u = tanh(T)
        return 1 - (u * u)
    return (E - e) * torch.reciprocal(E + e)


def celu(T, gradient=False, alpha=1.0):
    if gradient:
        return celu_derivative(T, alpha=alpha)
    tensor_positives = torch.gt(T, 0) * T
    tensor_negatives = torch.lt(T, 0) * T
    tensor_result = alpha * (torch.exp(torch.true_divide(tensor_negatives, alpha)) - 1)
    return tensor_result + tensor_positives


def celu_derivative(T, alpha=1.0):
    return torch.exp(T / alpha)


def sig(T, gradient=False):
    if gradient:
        sigT = sig(T)
        return sigT * (1 - sigT)
    return torch.reciprocal(1 + torch.exp(-1 * T))


def relu(T, gradient=False):
    if gradient:
        return torch.ge(T, 0) * torch.ones(T.size())
    return torch.ge(T, 0) * T


def swish(T, gradient=False, beta=1):
    if gradient:
        return swish_derivate(T, beta=beta)
    return T * sig(beta * T)


def swish_derivate(T, beta=1):
    return beta * swish(T, beta) + torch.matmul(sig(beta*T), 1 - beta * swish(T, beta))
