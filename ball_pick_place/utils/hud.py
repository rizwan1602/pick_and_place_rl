# SPDX-License-Identifier: Apache-2.0
"""
Interactive OpenCV HUD and Industrial Control Panel overlay for camera streams.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None


class CellCameraHUD:
    """Encapsulated OpenCV HUD and operator panel for robotic workcell camera feeds.
    
    Adheres to the Single Responsibility Principle (SRP) by decoupling UI event handling,
    touch targets, real-time perception overlay, and graphical rendering from the simulation loop.
    Eliminates global state mutation by managing event states internally.
    """

    # Button bounding boxes (x, y, w, h)
    BTN_START = (315, 14, 76, 40)
    BTN_STOP  = (396, 14, 72, 40)
    BTN_RESET = (473, 14, 76, 40)
    BTN_BALL  = (554, 14, 76, 40)

    def __init__(
        self,
        window_name: str = "Overhead 3D Camera - Industrial Digital Twin",
        width: int = 640,
        height: int = 480,
        enabled: bool = True,
    ) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height
        self.enabled = enabled and (cv2 is not None)
        self._pending_cmd: Optional[Union[str, Tuple[str, float, float]]] = None
        self._initialized = False

        if self.enabled:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.width, self.height)
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.setMouseCallback(self.window_name, self._on_mouse_event)
                self._initialized = True
            except Exception:
                self.enabled = False

    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Internal mouse callback for clickable HUD buttons and table coordinate unprojection."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        bx, by, bw, bh = self.BTN_START
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self._pending_cmd = "START"
            return

        bx, by, bw, bh = self.BTN_STOP
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self._pending_cmd = "STOP"
            return

        bx, by, bw, bh = self.BTN_RESET
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self._pending_cmd = "RESET"
            return

        bx, by, bw, bh = self.BTN_BALL
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self._pending_cmd = "SPAWN_BALL"
            return

        # Direct table click-to-place mapping
        if 70 <= y <= 450:
            norm_x = (x - 320) / 320.0
            norm_y = (y - 260) / 200.0
            target_y = float(np.clip(-norm_x * 0.22, -0.15, 0.15))
            target_x = float(np.clip(0.40 - norm_y * 0.16, 0.28, 0.44))
            self._pending_cmd = ("MOVE_BALL", target_x, target_y)

    def pop_command(self) -> Optional[Union[str, Tuple[str, float, float]]]:
        """Retrieve and clear the latest pending user command."""
        cmd = self._pending_cmd
        self._pending_cmd = None
        return cmd

    def render(
        self,
        rgb_tensor: torch.Tensor,
        plc_state: str = "IDLE",
        fsm_state: str = "RL_POLICY",
    ) -> Optional[str]:
        """Render frame with industrial status bar, buttons, and handle keyboard shortcuts.

        Returns:
            Optional command string if a keyboard shortcut was pressed, or 'EXIT' on Q/Esc.
        """
        if not self.enabled or not self._initialized:
            return None

        # Convert PyTorch RGB tensor to OpenCV BGR numpy array
        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = (rgb_tensor * 255).clamp(0, 255).to(torch.uint8)
        rgb_np = rgb_tensor.cpu().numpy()
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape

        # 1. Top Panel Bar (y: 0 to 68)
        cv2.rectangle(bgr, (0, 0), (w, 68), (24, 24, 24), -1)
        cv2.line(bgr, (0, 68), (w, 68), (55, 55, 55), 1)

        cv2.putText(bgr, "FRANKA PANDA - INDUSTRIAL PLC CELL", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        # 2. Glowing Green PLC Status LED
        cv2.circle(bgr, (20, 47), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(bgr, (20, 47), 10, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr, "PLC STATUS: ONLINE", (36, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

        # 3. State Pill
        if plc_state == "RUNNING":
            st_text = f"RUNNING [{fsm_state}]"
            st_col = (0, 255, 255)
        elif plc_state == "IDLE":
            st_text = "READY (Place ball -> Click START)"
            st_col = (0, 255, 140)
        elif plc_state == "STOPPED":
            st_text = "STOPPED (Arm Hold)"
            st_col = (0, 100, 255)
        elif plc_state == "RESETTING":
            st_text = "HOMING (Reset)"
            st_col = (255, 200, 0)
        else:
            st_text = plc_state
            st_col = (200, 200, 200)

        cv2.putText(bgr, f"| {st_text}", (195, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.38, st_col, 1, cv2.LINE_AA)

        # 4. START Button
        bx, by, bw, bh = self.BTN_START
        start_bg = (20, 75, 20) if plc_state == "RUNNING" else (16, 42, 16)
        start_border = (0, 255, 0) if plc_state == "RUNNING" else (0, 180, 0)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), start_bg, -1)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), start_border, 2)
        cv2.putText(bgr, "START", (bx + 10, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0) if plc_state != "RUNNING" else (255, 255, 255), 2, cv2.LINE_AA)

        # 5. STOP Button
        bx, by, bw, bh = self.BTN_STOP
        stop_bg = (20, 20, 80) if plc_state == "STOPPED" else (18, 18, 40)
        stop_border = (0, 0, 255) if plc_state == "STOPPED" else (0, 0, 180)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), stop_bg, -1)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), stop_border, 2)
        cv2.putText(bgr, "STOP", (bx + 12, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 60, 255) if plc_state != "STOPPED" else (255, 255, 255), 2, cv2.LINE_AA)

        # 6. RESET Button
        bx, by, bw, bh = self.BTN_RESET
        reset_bg = (70, 50, 15) if plc_state == "RESETTING" else (35, 28, 12)
        reset_border = (255, 200, 0) if plc_state == "RESETTING" else (180, 140, 0)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), reset_bg, -1)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), reset_border, 2)
        cv2.putText(bgr, "RESET", (bx + 10, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 50) if plc_state != "RESETTING" else (255, 255, 255), 2, cv2.LINE_AA)

        # 7. BALL Button
        bx, by, bw, bh = self.BTN_BALL
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (45, 25, 60), -1)
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (210, 120, 255), 2)
        cv2.putText(bgr, "BALL", (bx + 14, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 160, 255), 2, cv2.LINE_AA)

        # 8. Bottom Shortcut Bar
        cv2.rectangle(bgr, (0, h - 28), (w, h), (18, 18, 18), -1)
        cv2.putText(bgr, "CONTROLS: [S]=Start  [Space]=Stop  [R]=Reset  [B]=Spawn Ball  | Click table to place ball", 
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 210, 170), 1, cv2.LINE_AA)

        # 9. Ball Visual Detection Overlay (HSV segmentation)
        hsv = cv2.cvtColor(bgr[68:h-28, :], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 25:
                (bx_circ, by_circ), br_circ = cv2.minEnclosingCircle(c)
                bx_circ, by_circ = int(bx_circ), int(by_circ) + 68
                cv2.circle(bgr, (bx_circ, by_circ), int(br_circ) + 5, (0, 255, 0), 2)
                cv2.drawMarker(bgr, (bx_circ, by_circ), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                cv2.putText(bgr, "BALL TARGET", (bx_circ + 14, by_circ - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, bgr)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            return "EXIT"
        if key in (ord('s'), ord('S')):
            return "START"
        if key == 32:  # Space
            return "STOP"
        if key in (ord('r'), ord('R')):
            return "RESET"
        if key in (ord('b'), ord('B')):
            return "SPAWN_BALL"

        return None

    def close(self) -> None:
        """Destroy OpenCV window and release resources."""
        if self.enabled and self._initialized:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._initialized = False
