from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.topo import WorkerState
    from flight.nn.typing import Loss


class TrainerStrategy(t.Protocol):
    def trainer_kwargs(self) -> dict[str, t.Any]:
        pass

    def before_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        pass

    def after_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        pass
