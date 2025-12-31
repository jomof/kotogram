"""Training history management."""

import csv
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type, TypeVar

# pylint: disable=too-few-public-methods

T = TypeVar("T", bound="HistoryEvent")


class HistoryEvent(ABC):
    """Base class for history events."""

    @abstractmethod
    def to_row(self) -> List[str]:
        """Convert event to TSV row."""

    @classmethod
    @abstractmethod
    def from_row(cls: Type[T], row: List[str]) -> T:
        """Create event from TSV row."""

    @staticmethod
    @abstractmethod
    def get_type_name() -> str:
        """Get the unique type name for this event class."""


@dataclass
class KcEpochEvent(HistoryEvent):
    """Record of a completed KC pretraining epoch."""

    epoch: int
    metrics: Dict[str, Any]

    def to_row(self) -> List[str]:
        return [str(self.epoch), json.dumps(self.metrics)]

    @classmethod
    def from_row(cls, row: List[str]) -> "KcEpochEvent":
        if len(row) < 2:
            raise ValueError(f"KcEpochEvent row too short: {row}")
        return cls(
            epoch=int(row[0]),
            metrics=json.loads(row[1]),
        )

    @staticmethod
    def get_type_name() -> str:
        return "KC_EPOCH"


@dataclass
class StyleEpochEvent(HistoryEvent):
    """Record of a completed Style training epoch."""

    epoch: int
    metrics: Dict[str, Any]

    def to_row(self) -> List[str]:
        return [str(self.epoch), json.dumps(self.metrics)]

    @classmethod
    def from_row(cls, row: List[str]) -> "StyleEpochEvent":
        if len(row) < 2:
            raise ValueError(f"StyleEpochEvent row too short: {row}")
        return cls(
            epoch=int(row[0]),
            metrics=json.loads(row[1]),
        )

    @staticmethod
    def get_type_name() -> str:
        return "STYLE_EPOCH"


@dataclass
class KcDiagEvent(HistoryEvent):
    """Record of KC diagnostics for an epoch."""

    epoch: int
    stats: Dict[str, Any]

    def to_row(self) -> List[str]:
        return [str(self.epoch), json.dumps(self.stats)]

    @classmethod
    def from_row(cls, row: List[str]) -> "KcDiagEvent":
        if len(row) < 2:
            raise ValueError(f"KcDiagEvent row too short: {row}")
        return cls(
            epoch=int(row[0]),
            stats=json.loads(row[1]),
        )

    @staticmethod
    def get_type_name() -> str:
        return "KC_DIAG"


# Registry for event types
_EVENT_TYPES: Dict[str, Type[HistoryEvent]] = {
    KcEpochEvent.get_type_name(): KcEpochEvent,
    StyleEpochEvent.get_type_name(): StyleEpochEvent,
    KcDiagEvent.get_type_name(): KcDiagEvent,
}


def clear_history(file_path: str) -> None:
    """Clear the history file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)


def append_event(file_path: str, event: HistoryEvent) -> None:
    """Append an event to the history file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        # csv.writer with delimiter tab
        writer = csv.writer(f, delimiter="\t")
        row = [event.get_type_name()] + event.to_row()
        writer.writerow(row)


def read_events(file_path: str) -> List[HistoryEvent]:
    """Read all events from the history file."""
    if not os.path.exists(file_path):
        return []

    events = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            type_name = row[0]
            data = row[1:]

            event_cls = _EVENT_TYPES.get(type_name)
            if event_cls:
                events.append(event_cls.from_row(data))
            # Unknown types are skipped to allow forward compatibility

    return events
