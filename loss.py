import torch


def CE(p, q, estable=True, epsilon=1e-8):
    if estable:
        q = torch.clamp(q, min=epsilon)

    return -1 * torch.mul(p, torch.log(q)).sum()


def CELoss(Q, P, estable=True, epsilon=1e-8):
    # Q y P: representan distribuciones de probabilidad discreta  
    #        (mediante matrices con las mismas dimensiones)
    # estable y epsilon: nos permiten lograr estabilidad numérica cuando 
    #       intentamos computar el logaritmo de valores muy pequeños.
    #       epsilon limitará el valor mínimo del valor original cuando estable=True

    N = Q.size()[0]
    loss = 0.0

    for i in range(N):
        loss += CE(P[i], Q[i], estable, epsilon)

    return torch.true_divide(loss, N)

