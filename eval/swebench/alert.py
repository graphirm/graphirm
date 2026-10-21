"""Eval alerts are off.

Phone and email notifications were removed. `notify` stays so an old import
does not post anywhere. The watchdog restarts a dead T1 driver on its own.
"""

from __future__ import annotations


def notify(title: str, message: str, priority: str = "default") -> bool:
    del title, message, priority
    return False
