import torch.nn.functional as F


def loss_fn(logits, target):
    B, S, C = logits.shape
    # TODO: implement cross entropy directly
    return F.cross_entropy(logits.view(B * S, C), target.view(-1))


def grad_norm(model):
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total**0.5


def weight_norm(model):
    total = 0
    for p in model.parameters():
        total += p.norm().item() ** 2
    return total**0.5
