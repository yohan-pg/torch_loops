import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_loops import TrainingLoop, EvaluationLoop


def test_training_loop(tolerence=0.1, num_batches=1000, batch_size=3):
    for amp in [True, False]:
        inputs = torch.randn(batch_size, 2)
        targets = torch.randn(batch_size, 2)

        model = nn.Linear(2, 2)
        training = TrainingLoop(
            model,
            dataloader=iter([(inputs, targets)] * num_batches),
            criterions=dict(mse=(1.0, nn.MSELoss()), l1=(0.5, nn.L1Loss())),
            optimizers=[torch.optim.Adam(model.parameters(), lr=1e-1)],
            amp=amp
        )

        i = 0
        with torch.no_grad():
            old_preds = model(inputs)

        for i, (preds, losses) in enumerate(training):
            # Autograd is enabled in the loop scope
            assert torch.randn(2, requires_grad=True).square().grad_fn is not None
            
            # The prediction shapes are OK
            assert preds.shape == targets.shape

            # The predictions returned match are the predictions right before the step
            assert preds.allclose(old_preds)
            with torch.no_grad():
                old_preds = model(inputs)

            # The predictions change
            with torch.no_grad():
                assert not preds.allclose(model(inputs))
            
            # The loss shapes are OK
            for loss in losses.values():
                assert loss.shape == () 

            # The losses include the final weighted objective
            assert set(losses.keys()) == set(training.criterions.keys()) | set(["objective"])

            # The losses are measured correctly
            assert losses["mse"].allclose(F.mse_loss(preds, targets))
            assert losses["l1"].allclose(F.l1_loss(preds, targets))
            assert losses["objective"].allclose(1.0 * losses["mse"] + 0.5 * losses["l1"])

            # Optimization converges
            if i == num_batches - 1:
                assert losses["objective"] < tolerence
                assert losses["mse"] < tolerence
                assert losses["l1"] < tolerence
                
        # Trainings passes over all batches onces
        assert i == num_batches - 1 


# todo test that grad is disabled

def test_inference_loop(num_batches=1000):
    for amp in [True, False]:
        inputs = [torch.randn(2, 2) for _ in range(num_batches)]
        targets = [torch.randn(2, 2) for _ in range(num_batches)]

        model = nn.Linear(2, 2)
        
        evaluation = EvaluationLoop(
            model,
            dataloader=zip(inputs, targets),
            metrics=dict(mse=nn.MSELoss(), l1=nn.L1Loss()),
            amp=amp
        )

        i = 0 
        for i, (preds, scores) in enumerate(evaluation):
            # Autograd is disabled in the loop scope
            assert torch.randn(2, requires_grad=True).square().grad_fn is None

            # The model is set to eval mode
            assert not model.training

            # The computations were performed without autograd
            assert not preds.requires_grad

            # The predictions are computed correctly
            assert preds.shape == targets[0].shape
            assert preds.allclose(model(inputs[i]))

            # The score shapes are ok
            assert set(scores.keys()) == set(["mse", "l1"])
            assert scores["mse"].shape == ()
            assert scores["l1"].shape == ()

            # The scores do not have autograd
            assert scores["mse"].grad_fn is None
            assert scores["l1"].grad_fn is None

            # The scores are measured correctly
            assert scores["mse"].allclose(F.mse_loss(model(inputs[i]), targets[i]))
            assert scores["l1"].allclose(F.l1_loss(model(inputs[i]), targets[i]))
        
        assert model.training

        # Evaluation passes over all batches onces
        assert i == num_batches - 1