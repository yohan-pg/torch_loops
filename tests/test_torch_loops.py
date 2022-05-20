import torch
from torch_loops import training_loop, evaluation_loop, aggregate


def test_training_loop(tolerence=0.1, num_batches=1000):
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()

    for i, (y_hat, losses) in enumerate(
        training_loop(
            # todo this is wrong, should be model(x), but the type hint is wrong
            model=model.forward,
            dataloader=iter([(x, y)] * num_batches),
            criterions=dict(mse=criterion),
            optimizer=opt,
        )
    ):
        assert y_hat.shape == y.shape
        assert losses["mse"].shape == ()
        assert losses["total"].shape == ()

        if i == num_batches - 1:
            assert criterion(y_hat, y).item() < tolerence
            return
    assert False


def test_inference_loop(num_batches=1000):
    xs = [torch.randn(2, 2) for _ in range(num_batches)]
    ys = [torch.randn(2, 2) for _ in range(num_batches)]

    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.MSELoss()

    for i, (y_hat, losses) in enumerate(
        evaluation_loop(
            # todo this is wrong, should be model(x), but the type hint is wrong
            dataloader=zip(xs, ys),
            model=model.forward,
            criterions=dict(mse=criterion),
        )
    ):
        assert not y_hat.requires_grad
        assert y_hat.shape == ys[0].shape
        assert losses["mse"].shape == ()

        if i == num_batches - 1:
            return
    assert False


def test_aggregate(num_batches=1000, window_size=10, batch_size=3):
    xs = [torch.randn(batch_size, 2) for _ in range(num_batches)]
    ys = [torch.randn(batch_size, 2) for _ in range(num_batches)]

    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.MSELoss()

    loop = evaluation_loop(
        # todo this is wrong, should be model(x), but the type hint is wrong
        dataloader=zip(xs, ys),
        model=model.forward,
        criterions=dict(mse=criterion),
    )

    preds_stats, losses_stats = aggregate(loop, window_size=window_size)
    
    assert len(preds_stats) == num_batches // window_size
    assert len(losses_stats) == num_batches // window_size

    assert len(preds_stats[0]) == 1
    assert preds_stats[0][0].shape == (batch_size * window_size, 2)

    assert len(losses_stats[0]) == 1
    assert len(losses_stats[0]["mse"]) == 2
    assert losses_stats[0]["mse"][0].shape == ()
    assert losses_stats[0]["mse"][1].shape == ()
