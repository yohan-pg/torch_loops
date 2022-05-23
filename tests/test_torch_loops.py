import torch
from torch_loops import training_loop, evaluation_loop


def test_training_loop(tolerence=0.1, num_batches=1000):
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()
    criterions = dict(mse=(1.0, criterion))
    training = training_loop(
        # todo this is wrong, should be model(x), but the type hint is wrong
        model=model.forward,
        dataloader=iter([(x, y)] * num_batches),
        criterions=criterions,
        optimizers=[opt],
    )

    for i, losses in enumerate(training()):
        assert set(losses.columns) == set(criterions.keys()) | set(["objective"])

        if i == 0:
            assert losses.shape == (1, 2)

        if i == num_batches - 1:
            assert losses.shape == (num_batches, 2)
            assert criterion(model(x), y).item() < tolerence
            return

    #! fails to test ending
    assert False

    # todo test the weighting of the losses


def test_inference_loop(num_batches=1000):
    xs = [torch.randn(2, 2) for _ in range(num_batches)]
    ys = [torch.randn(2, 2) for _ in range(num_batches)]

    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.MSELoss()
    evaluation = evaluation_loop(
        # todo this is wrong, should be model(x), but the type hint is wrong
        dataloader=zip(xs, ys),
        model=model.forward,
        metrics=dict(mse=criterion),
    )

    for i, (y_hat, losses) in enumerate(evaluation()):
        assert not y_hat.requires_grad
        assert y_hat.shape == ys[0].shape

        if i == 0:
            assert losses["mse"].shape == (1,)

        if i == num_batches - 1:
            assert losses["mse"].shape == (num_batches,)
            return

    #! fails to test ending
    assert False
