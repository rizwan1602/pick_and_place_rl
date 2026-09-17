# SPDX-License-Identifier: Apache-2.0
'''
Shared State Bridge between Siemens PLC Extension UI, OPC UA Manager, and Franka Pick-Place Policy.
'''

from __future__ import annotations
from typing import Callable, Optional, Tuple, Any

class RobotCellBridge:
    '''Singleton bridge connecting Siemens PLC Bridge Extension UI to Franka RL Controller.'''
    _instance: Optional['RobotCellBridge'] = None

    @classmethod
    def get(cls) -> 'RobotCellBridge':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.pending_cmd: Optional[Any] = None
        self.robot_state: str = "IDLE"  # IDLE, RUNNING, STOPPED, RESETTING
        self.plc_online: bool = True     # Status indicator green
        self.ball_x: float = 0.35
        self.ball_y: float = 0.00
        self.cycle_count: int = 0
        self.status_message: str = "System Ready. Place ball and click START."
        self._listeners: list[Callable[['RobotCellBridge'], None]] = []

    def register_listener(self, cb: Callable[['RobotCellBridge'], None]):
        if cb not in self._listeners:
            self._listeners.append(cb)

    def notify(self):
        for cb in self._listeners:
            try:
                cb(self)
            except Exception:
                pass

    def trigger_start(self):
        self.pending_cmd = "START"
        self.robot_state = "RUNNING"
        self.status_message = "START command issued -> Pick & Place active"
        self.notify()

    def trigger_stop(self):
        self.pending_cmd = "STOP"
        self.robot_state = "STOPPED"
        self.status_message = "STOP command issued -> Robot halted"
        self.notify()

    def trigger_reset(self):
        self.pending_cmd = "RESET"
        self.robot_state = "RESETTING"
        self.status_message = "RESET command issued -> Returning to Home Standby"
        self.notify()

    def trigger_move_ball(self, x: float, y: float):
        self.ball_x = float(x)
        self.ball_y = float(y)
        self.pending_cmd = ("MOVE_BALL", self.ball_x, self.ball_y)
        self.status_message = f"Ball positioned at X={self.ball_x:.3f}m, Y={self.ball_y:.3f}m"
        self.notify()

    def trigger_spawn_ball(self):
        self.pending_cmd = "SPAWN_BALL"
        self.status_message = "Ball spawned at randomized table position"
        self.notify()

    def pop_command(self) -> Optional[Any]:
        cmd = self.pending_cmd
        self.pending_cmd = None
        return cmd
