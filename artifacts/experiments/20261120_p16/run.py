"""Train P1.6 to the full budget, then queue the checkpoint-selected evaluation in the same service (P1.5's one-shot
worker). The evaluation module is written during training in this folder; the trainer and this queue are sealed by
SOURCE_PINS.json (saccade/ must not change while it runs)."""
import importlib
from types import SimpleNamespace

from . import train as launch

WORKER = importlib.import_module("artifacts.experiments.20261104_p1f21.run")
L = SimpleNamespace(recipe=lambda _cfg=None: launch.recipe(), check_recipe=launch.check_recipe,
                    WORLD_SEED=launch.WORLD_SEED, C=launch.F61.C)


if __name__ == "__main__":
    WORKER.main(L=L, here=launch.HERE, module=f"{__package__}.train", gates=("checks.json",),
                evaluation_module=f"{__package__}.evaluation")
