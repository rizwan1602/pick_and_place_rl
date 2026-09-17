# SPDX-License-Identifier: Apache-2.0
"""
Filtering and signal processing utilities for robotic controllers.
"""

from __future__ import annotations
from typing import Optional
import torch


class ActionSmoother:
    """Exponential Moving Average (EMA) action filter for continuous robotic policies.

    Mitigates high-frequency policy jitter (e.g. 50-100 Hz micro-vibrations) by
    smoothing control actions in normalized action space [-1.0, 1.0].
    
    Formula:
        a_exec(t) = alpha * a_exec(t-1) + (1 - alpha) * a_raw(t)

    Attributes:
        alpha: Smoothing coefficient in [0.0, 1.0].
            Higher alpha yields smoother trajectories; lower alpha yields faster reaction.
            Standard default is 0.65 for Franka Panda dynamics.
    """

    def __init__(self, alpha: float = 0.65) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0.0, 1.0], received: {alpha}")
        self.alpha: float = alpha
        self.filtered_action: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset internal filter state."""
        self.filtered_action = None

    def filter(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply EMA smoothing to a raw action tensor.

        Args:
            raw_action: PyTorch tensor containing raw policy output deltas.

        Returns:
            Smoothed action tensor with identical dimensions and device placement.
        """
        if self.filtered_action is None:
            self.filtered_action = raw_action.clone()
        else:
            self.filtered_action = self.alpha * self.filtered_action + (1.0 - self.alpha) * raw_action
        return self.filtered_action
