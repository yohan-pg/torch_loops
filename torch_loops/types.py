from prelude import *

Inputs = TypeVar("Inputs")
Preds = TypeVar("Preds")
Targets = TypeVar("Targets")

Loss = torch.Tensor
Losses = dict[str, Loss]

# Scores are losses which do not require differentiability
Score = float
Scores = dict[str, Score]

Batch = tuple[Preds, Losses]

Weight = float
Criterion = Callable[[Preds, Targets], Loss]
Metric = Callable[[Preds, Targets], Score]
Model = Callable[[Inputs], Preds]