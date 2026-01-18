# pylint: disable=duplicate-code
import dataclasses
from typing import Any, Dict, List
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from kotogram.model import NUM_REGISTER_CLASSES
from train.config import HardwareConfig, TrainerConfig
from train.trainer import Trainer
from train.trainer_view import TrainerView
from train.types import EvaluationMetrics, TrainingHistory


def create_dummy_trainer_config():
    return TrainerConfig(
        device="cpu",
        batch_size=2,
        epochs=1,
        hardware=HardwareConfig(cpu_reserve_cores=1),
    )


def create_tiny_style_dataset():
    mock = MagicMock()
    mock.__len__.return_value = 10

    sample = MagicMock()
    sample.feature_ids = {
        "surface": [1, 2],
        "lemma": [1, 2],
        "reading": [1, 2],
        "pos": [1, 2],
        "base_form": [1, 2],
        "c_type": [1, 2],
        "c_form": [1, 2],
        "normalized_surface": [1, 2],
        "full_reading": [1, 2],
        "pronunciation": [1, 2],
        "base_orth": [1, 2],
        "reading_gram": [1, 2],
    }
    sample.formality_value = 0.5
    sample.formality_pragmatic = 0
    sample.gender_value = 0.0
    sample.gender_pragmatic = 1
    sample.grammaticality_label = 1
    sample.register_labels = [0]
    sample.original_sentence = "test"
    sample.kotogram = "test"
    sample.kc_targets = {}
    sample.idx = 0

    mock.__getitem__.return_value = sample

    mock.get_formality_class_weights.return_value = torch.ones(5)
    mock.get_gender_class_weights.return_value = torch.ones(3)
    mock.get_grammaticality_class_weights.return_value = torch.ones(2)
    mock.get_register_class_weights.return_value = torch.ones(4)
    return mock


class DummyModel(torch.nn.Module):
    def __init__(self, config, _vocab_size):
        super().__init__()
        self.config = config
        self.encoder = torch.nn.Linear(1, 1)
        self.embedding = torch.nn.Linear(1, 1)
        self.formality_value_head = torch.nn.Linear(1, 1)
        self.formality_pragmatic_head = torch.nn.Linear(1, 1)
        self.gender_value_head = torch.nn.Linear(1, 1)
        self.gender_pragmatic_head = torch.nn.Linear(1, 1)
        self.grammaticality_head = torch.nn.Linear(1, 1)
        self.register_head = torch.nn.Linear(1, NUM_REGISTER_CLASSES)
        self.pooler = torch.nn.Linear(1, 1)
        self.grammaticality_classifier = self.grammaticality_head
        self.register_classifier = self.register_head

    def forward(self, *_args, **_kwargs):
        batch_size = 2
        return (
            torch.randn(batch_size, 1, requires_grad=True),  # f_val [B, 1]
            torch.randn(batch_size, 5, requires_grad=True),  # f_prag [B, 5]
            torch.randn(batch_size, 1, requires_grad=True),  # g_val [B, 1]
            torch.randn(batch_size, 3, requires_grad=True),  # g_prag [B, 3]
            torch.randn(batch_size, 2, requires_grad=True),  # gram [B, 2]
            torch.randn(
                batch_size, NUM_REGISTER_CLASSES, requires_grad=True
            ),  # reg [B, NUM_REGISTER_CLASSES]
        )


@dataclasses.dataclass
class RecordedCall:
    name: str
    args: Dict[str, Any]


class RecordingTrainerView(TrainerView):
    def __init__(self):
        self.calls: List[RecordedCall] = []

    def _record(self, event_name: str, **kwargs):
        self.calls.append(RecordedCall(name=event_name, args=kwargs))

    def on_train_start(self, epochs: int, start_epoch: int, start_batch: int) -> None:
        self._record(
            "on_train_start",
            epochs=epochs,
            start_epoch=start_epoch,
            start_batch=start_batch,
        )

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self._record("on_epoch_start", epoch=epoch, total_epochs=total_epochs)

    # pylint: disable=too-many-positional-arguments
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: tuple[float, float, float, float, float],
        eval_metrics: EvaluationMetrics,
        avg_acc: float,
        is_best: bool,
        patience_counter: int,
    ) -> None:
        self._record(
            "on_epoch_end",
            epoch=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            avg_acc=avg_acc,
            is_best=is_best,
            patience_counter=patience_counter,
        )

    def on_train_end(self, history: TrainingHistory) -> None:
        self._record("on_train_end", history=history)

    def on_progress_init(self, desc: str, total_steps: int) -> None:
        self._record("on_progress_init", desc=desc, total_steps=total_steps)

    def on_progress_update(self, batch_idx: int, loss: float, total_steps: int) -> None:
        self._record(
            "on_progress_update",
            batch_idx=batch_idx,
            loss=loss,
            total_steps=total_steps,
        )

    def on_progress_stop(self) -> None:
        self._record("on_progress_stop")

    def on_timing_summary(
        self,
        avg_total_ms: float,
        avg_data_ms: float,
        avg_compute_ms: float,
        data_frac: float,
    ) -> None:
        self._record(
            "on_timing_summary",
            avg_total_ms=avg_total_ms,
            avg_data_ms=avg_data_ms,
            avg_compute_ms=avg_compute_ms,
            data_frac=data_frac,
        )

    def on_best_model_saved(self, model_path: str, best_val_loss: float) -> None:
        self._record(
            "on_best_model_saved", model_path=model_path, best_val_loss=best_val_loss
        )

    def on_warning(self, message: str) -> None:
        self._record("on_warning", message=message)

    def on_early_stopping(self, epoch: int) -> None:
        self._record("on_early_stopping", epoch=epoch)

    def on_line_flush(self) -> None:
        self._record("on_line_flush")

    def on_style_epoch_eval_stats(
        self, epoch: int, stats: List[Dict[str, Any]]
    ) -> None:
        self._record("on_style_epoch_eval_stats", epoch=epoch, stats=stats)


class TestTrainerView(TestCase):
    def setUp(self):
        self.config = create_dummy_trainer_config()
        self.dataset = create_tiny_style_dataset()
        self.model = DummyModel(self.config, 100)
        self.view = RecordingTrainerView()

        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.dataset,
            val_dataset=self.dataset,
            config=self.config,
            dl_config_train=self.config.resolve_dataloader_config(
                torch.device("cpu"), "train"
            ),
            dl_config_val=self.config.resolve_dataloader_config(
                torch.device("cpu"), "val"
            ),
            output_path="/tmp/output",
            view=self.view,
        )

    def test_train_calls_view_hooks(self):
        # Mock save_model to avoid size verification on DummyModel
        # Since train_io is imported in trainer.py, checking where to patch
        # But we can just patch 'train.trainer.train_io.save_model' or mock the method
        # Actually easier to use unittest.mock.patch
        # Since Trainer imports save_model directly: `from train.io import save_model`
        # We must patch `train.trainer.save_model` to affect the Trainer.
        from unittest.mock import patch

        with patch("train.style_trainer.save_model"):
            self.trainer.evaluate = MagicMock(return_value=EvaluationMetrics(loss=0.5))

            self.trainer.train(
                epochs=1,
                on_epoch_end=lambda h: None,
            )

        calls = [c.name for c in self.view.calls]

        self.assertIn("on_train_start", calls)
        self.assertIn("on_epoch_start", calls)
        self.assertIn("on_progress_init", calls)
        self.assertIn("on_progress_update", calls)
        self.assertIn("on_progress_stop", calls)
        self.assertIn("on_epoch_end", calls)
        self.assertIn("on_train_end", calls)
        self.assertIn("on_timing_summary", calls)
