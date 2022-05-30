from .prelude import *
from .types import *

@dataclass
class TrainingLoop(Generic[Inputs, Preds, Targets]):
    model: Model
    _ = dataclasses.KW_ONLY
    dataloader: Iterable[tuple[Inputs, Targets]]
    criterions: dict[str, tuple[Weight, Criterion]]
    optimizers: Iterable[torch.optim.Optimizer]
    schedulers: Iterable[torch.optim.lr_scheduler._LRScheduler] = ()
    amp: bool = True
    seed: int = 0

    _scaler: Optional[GradScaler] = field(default=None, repr=False)

    def __post_init__(self):
        self._scaler = GradScaler() if self.amp else None
        self.last_epoch_losses = None

    def step(self, batch: Batch) -> tuple[Preds, dict[str, Loss]]:
        inputs, targets = batch

        preds = self.model(inputs)

        losses = {
            label: criterion(preds, targets)
            for label, (_, criterion) in self.criterions.items()
        }

        objective = torch.tensor(0.0)
        for label, loss in losses.items():
            weight = self.criterions[label][0]
            objective += weight * loss.mean(dim=0)

        return preds, (losses | {"objective": objective})

    def evaluation(self, batch: Batch):
        return self.step(batch)[1]

    def __iter__(self) -> Iterator[tuple[Preds, Losses]]:
        torch.manual_seed(self.seed)

        epoch_losses = { label: 0.0 for label in self.criterions.keys() }

        i = 0
        for batch in self.dataloader:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            with autocast() if self.amp else nullcontext():
                preds, losses = self.step(batch)

            if self._scaler is not None:
                self._scaler.scale(losses["objective"]).backward()  # type: ignore
            else:
                losses["objective"].backward()

            for optimizer in self.optimizers:
                if self._scaler is not None:
                    self._scaler.step(optimizer)  # type: ignore
                else:
                    optimizer.step()

            for scheduler in self.schedulers:
                scheduler.step()

            if self._scaler is not None:
                self._scaler.update()

            for key in self.criterions.keys():
                epoch_losses[key] += losses[key].item() / len(self)

            yield preds, losses

            i += 1

        self.last_epoch_losses = epoch_losses

    def __len__(self) -> int:
        return len(self.dataloader)


@dataclass
class EvaluationLoop(Generic[Inputs, Preds, Targets]):
    model: Model
    _ = dataclasses.KW_ONLY
    dataloader: Iterable[Batch]
    metrics: dict[str, Callable[[Preds, Targets], Score]]
    amp: bool = True
    seed: int = 0

    def __iter__(self) -> Iterator[tuple[Preds, Scores]]:
        torch.manual_seed(self.seed)

        with torch.inference_mode():
            was_training = self.model.training
            self.model.train(False)

            for inputs, targets in self.dataloader:
                with autocast() if self.amp else nullcontext():
                    preds = self.model(inputs)
                    scores = {
                        label: criterion(preds, targets).item()
                        for label, criterion in self.metrics.items()
                    }

                    yield preds, scores

            self.model.train(was_training)

    def __len__(self) -> int:
        return len(self.dataloader)