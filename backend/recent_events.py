"""Helpers for keeping a short history of ADA events."""

from typing import Any


def remember_event(
    event: dict[str, Any],
    history: list[dict[str, Any]] = [],
) -> list[dict[str, Any]]:
    """Add an event to the in-memory history."""
    history.append(event)
    return history


def average_confidence(events: list[dict[str, Any]]) -> float:
    """Return the average confidence across recent events."""
    total = sum(float(event["confidence"]) for event in events)
    return total / len(events)
