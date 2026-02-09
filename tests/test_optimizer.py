import torch

from .adapters import get_sgd_cls

def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()

    return model.weight.detach()

def test_sgd():
    pytorch_weights = _optimize(torch.optim.SGD)
    actual_weights = _optimize(get_sgd_cls())

    assert torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
