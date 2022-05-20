import torch
import torch.nn as nn
from typing import *  # type: ignore
from torch.utils.data import DataLoader

from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import itertools
import contextlib


Inputs = TypeVar("Inputs", bound=dict[str, torch.Tensor] | torch.Tensor)
Preds = TypeVar("Preds", bound=dict[str, torch.Tensor] | torch.Tensor)
Targets = TypeVar("Targets", bound=dict[str, torch.Tensor] | torch.Tensor)

Loss = torch.Tensor
Losses = dict[str, Loss]

Metric = torch.Tensor  # Metrics are losses which do not require differentiability
Metrics = dict[str, torch.Tensor]


def training_loop(
    *,
    model: Callable[[Inputs], Preds],
    dataloader: Iterator[tuple[Inputs, Targets]],  # type: ignore
    criterions: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    amp: bool = False,
) -> Iterator[tuple[Preds, Losses]]:
    if amp:
        scaler = GradScaler()
    else:
        scaler = None

    for inputs, targets in dataloader:
        with autocast() if amp else contextlib.nullcontext():
            preds = model(inputs)
            losses = {
                key: criterion(preds, targets) for key, criterion in criterions.items()
            }
            total_loss = sum(
                [value.mean(dim=0) for value in losses.values()], torch.tensor(0.0)
            )

        optimizer.zero_grad()

        if amp:
            assert scaler is not None
            scaler.scale(total_loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # todo where to place this yield statement exactly?
        yield preds, losses | {"total": total_loss}


@torch.inference_mode()
def evaluation_loop(
    *,
    model: Callable[[Inputs], Preds],
    dataloader: Iterator[Tuple[Inputs, Targets]],
    criterions: dict[str, Callable[[Preds, Targets], Metric]],
    amp: bool = False,
) -> Iterator[tuple[Preds, Metrics]]:
    for inputs, targets in dataloader:
        with autocast() if amp else contextlib.nullcontext():
            preds = model(inputs)
            losses = {
                key: criterion(preds, targets) for key, criterion in criterions.items()
            }

        yield preds, losses


def aggregate(
    loop,
    *,
    window_size: int,
    pred_transforms: Optional[list[Callable]] = [lambda x: x.flatten(0, 1)],
    loss_transforms: Optional[list[Callable]] = [torch.mean, torch.std],
):
    preds_agg = []
    preds_window = []

    losses_agg = []
    losses_window = []

    def step(collection, window, agg, transforms):
        if transforms is not None:
            window.append(collection)

            if i % window_size == window_size - 1:
                if isinstance(collection, dict):
                    window_transposed = {
                        label: torch.stack([values[label] for values in window])
                        for label in collection.keys()
                    }
                    agg.append(
                        {
                            key: [transform(stack) for transform in transforms]
                            for key, stack in window_transposed.items()
                        }
                    )
                    window.clear()
                else:
                    agg.append(
                        [transform(torch.stack(window)) for transform in transforms]
                    )
                    window.clear()

    for i, (preds, losses) in enumerate(loop):
        step(preds, preds_window, preds_agg, pred_transforms)
        step(losses, losses_window, losses_agg, loss_transforms)

    return preds_agg, losses_agg
