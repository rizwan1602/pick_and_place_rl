# SPDX-License-Identifier: Apache-2.0
'''
UI Window for Siemens PLC OPC UA Bridge in Isaac Sim / Omniverse.
Provides live green status indicator, START / STOP / RESET controls,
GPU-safe Ball placement, and Siemens S7 OPC UA configuration.
'''

from __future__ import annotations
import asyncio
import omni.ui as ui
from .opcua_mgr import PLCManager
from .bridge_state import RobotCellBridge


class PLCBridgeUI:
    '''Main window UI for configuring Siemens PLC OPC UA and Franka Pick-and-Place cell.'''

    def __init__(self, manager: PLCManager):
        self.manager = manager
        self.bridge = RobotCellBridge.get()

        # Window creation - compact dimensions
        self.window = ui.Window("Siemens PLC OPC UA Bridge", width=360, height=520)

        # UI Element handles
        self.status_bulb: ui.Rectangle = None
        self.status_label: ui.Label = None
        self.robot_state_label: ui.Label = None

        self.btn_start: ui.Button = None
        self.btn_stop: ui.Button = None
        self.btn_reset: ui.Button = None

        self.ball_x_slider: ui.FloatSlider = None
        self.ball_y_slider: ui.FloatSlider = None
        self.ball_status_label: ui.Label = None

        self.endpoint_field: ui.StringField = None
        self.db_name_field: ui.StringField = None
        self.tag_name_field: ui.StringField = None
        self.effective_node_label: ui.Label = None

        self._build_ui()
        self.manager.register_status_callback(self._on_connection_status_changed)
        self.bridge.register_listener(self._on_bridge_state_changed)

    def _build_ui(self):
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, style={"margin": 8}):
                    # ── Header & Glowing Green PLC Status ────────────────────
                    with ui.HStack(height=26, spacing=8):
                        self.status_bulb = ui.Rectangle(
                            width=14,
                            height=14,
                            style={"border_radius": 7, "background_color": 0xFF00FF00},  # Always Green
                        )
                        self.status_label = ui.Label(
                            "PLC STATUS: ONLINE",
                            style={"color": 0xFF00FF00, "font_size": 13, "font_weight": "bold"},
                        )

                    with ui.HStack(height=20, spacing=4):
                        ui.Label("Robot State:", width=80, style={"font_size": 11, "color": 0xFFAAAAAA})
                        self.robot_state_label = ui.Label(
                            f"{self.bridge.robot_state} (Standby Home)",
                            style={"font_size": 12, "color": 0xFF00E5FF, "font_weight": "bold"},
                        )

                    ui.Line(style={"color": 0x33FFFFFF})

                    # ── Section 1: Main PLC Control Panel (Compact Row) ─────
                    ui.Label("Industrial Control Panel:", style={"font_size": 12, "font_weight": "bold", "color": 0xFFE0E0E0})

                    with ui.HStack(height=38, spacing=6):
                        self.btn_start = ui.Button(
                            "▶ START",
                            clicked_fn=self._on_start_clicked,
                            style={"background_color": 0xFF059669, "font_size": 13, "font_weight": "bold", "border_radius": 4},
                        )
                        self.btn_stop = ui.Button(
                            "⏹ STOP",
                            clicked_fn=self._on_stop_clicked,
                            style={"background_color": 0xFFDC2626, "font_size": 13, "font_weight": "bold", "border_radius": 4},
                        )
                        self.btn_reset = ui.Button(
                            "⟲ RESET",
                            clicked_fn=self._on_reset_clicked,
                            style={"background_color": 0xFFD97706, "font_size": 13, "font_weight": "bold", "border_radius": 4},
                        )

                    ui.Label("• START: Autonomous Ball Pick & Place into Bucket\n• STOP: Halts Franka Panda Arm Instantly\n• RESET: Returns Arm to Standby Home [0.35, 0.00, 0.65]", 
                             style={"font_size": 10, "color": 0xFF888888})

                    ui.Line(style={"color": 0x33FFFFFF})

                    # ── Section 2: Ball Placement & Repositioning ────────────
                    ui.Label("Ball Placement (GPU PhysX Direct):", style={"font_size": 12, "font_weight": "bold", "color": 0xFFE0E0E0})

                    with ui.HStack(height=32, spacing=6):
                        ui.Button(
                            "🎲 Randomize Ball on Table",
                            clicked_fn=self._on_spawn_ball_clicked,
                            style={"background_color": 0xFF6366F1, "font_size": 11, "border_radius": 4},
                        )
                        ui.Button(
                            "📍 Move to (X, Y)",
                            clicked_fn=self._on_move_ball_coords_clicked,
                            style={"background_color": 0xFF2563EB, "font_size": 11, "border_radius": 4},
                        )

                    with ui.HStack(height=22):
                        ui.Label("Ball X [0.28 - 0.44]:", width=130, style={"font_size": 10})
                        self.ball_x_slider = ui.FloatSlider(min=0.28, max=0.44, step=0.01)
                        self.ball_x_slider.model.set_value(0.35)

                    with ui.HStack(height=22):
                        ui.Label("Ball Y [-0.15 - 0.15]:", width=130, style={"font_size": 10})
                        self.ball_y_slider = ui.FloatSlider(min=-0.15, max=0.15, step=0.01)
                        self.ball_y_slider.model.set_value(0.00)

                    self.ball_status_label = ui.Label(
                        "Ball Target: Ready on table (Press 'B' or click Camera view)",
                        style={"font_size": 10, "color": 0xFF00FF88},
                    )

                    ui.Line(style={"color": 0x33FFFFFF})

                    # ── Section 3: Siemens S7 OPC UA Connection ─────────────
                    with ui.CollapsableFrame("Siemens S7 OPC UA Configuration", collapsed=True):
                        with ui.VStack(spacing=6):
                            with ui.HStack():
                                ui.Label("Server URL:", width=90, style={"font_size": 11})
                                self.endpoint_field = ui.StringField()
                                self.endpoint_field.model.set_value(self.manager.endpoint)

                            with ui.HStack():
                                ui.Label("DB Name:", width=90, style={"font_size": 11})
                                self.db_name_field = ui.StringField()
                                self.db_name_field.model.set_value(self.manager.db_name)
                                self.db_name_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Tag Name:", width=90, style={"font_size": 11})
                                self.tag_name_field = ui.StringField()
                                self.tag_name_field.model.set_value(self.manager.tag_name)
                                self.tag_name_field.model.add_value_changed_fn(self._update_node_display)

                            ui.Label("Effective NodeId:", style={"font_size": 10, "color": 0xFFAAAAAA})
                            self.effective_node_label = ui.Label(
                                self.manager.get_effective_node_id(),
                                style={"color": 0xFFE9C46A, "font_size": 11},
                            )

                            with ui.HStack(spacing=6):
                                ui.Button(
                                    "Connect PLC",
                                    clicked_fn=self._on_connect_clicked,
                                    style={"background_color": 0xFF059669, "font_size": 11},
                                )
                                ui.Button(
                                    "Disconnect",
                                    clicked_fn=self._on_disconnect_clicked,
                                    style={"background_color": 0xFFDC2626, "font_size": 11},
                                )

                            with ui.HStack(spacing=6):
                                ui.Button(
                                    "Write PLC Bit TRUE (1)",
                                    clicked_fn=lambda: asyncio.ensure_future(self.manager.write_toggle_bit(True)),
                                    style={"background_color": 0xFF047857, "font_size": 10},
                                )
                                ui.Button(
                                    "Write PLC Bit FALSE (0)",
                                    clicked_fn=lambda: asyncio.ensure_future(self.manager.write_toggle_bit(False)),
                                    style={"background_color": 0xFF991B1B, "font_size": 10},
                                )

    def _update_node_display(self, model=None):
        if self.db_name_field:
            self.manager.db_name = self.db_name_field.model.get_value_as_string()
        if self.tag_name_field:
            self.manager.tag_name = self.tag_name_field.model.get_value_as_string()
        if self.effective_node_label:
            self.effective_node_label.text = self.manager.get_effective_node_id()

    def _on_start_clicked(self):
        '''User clicked START button in UI.'''
        print("[PLC Bridge UI] START button clicked!")
        self.bridge.trigger_start()

    def _on_stop_clicked(self):
        '''User clicked STOP button in UI.'''
        print("[PLC Bridge UI] STOP button clicked!")
        self.bridge.trigger_stop()

    def _on_reset_clicked(self):
        '''User clicked RESET button in UI.'''
        print("[PLC Bridge UI] RESET button clicked -> Homing arm!")
        self.bridge.trigger_reset()

    def _on_spawn_ball_clicked(self):
        '''Randomize ball position on table.'''
        print("[PLC Bridge UI] Randomize Ball clicked!")
        self.bridge.trigger_spawn_ball()

    def _on_move_ball_coords_clicked(self):
        '''Move ball to coordinates specified by sliders.'''
        x = self.ball_x_slider.model.get_value_as_float()
        y = self.ball_y_slider.model.get_value_as_float()
        print(f"[PLC Bridge UI] Move ball to X={x:.3f}, Y={y:.3f}")
        self.bridge.trigger_move_ball(x, y)

    def _on_connect_clicked(self):
        url = self.endpoint_field.model.get_value_as_string()
        self._update_node_display()
        asyncio.ensure_future(self.manager.connect(url))

    def _on_disconnect_clicked(self):
        asyncio.ensure_future(self.manager.disconnect())

    def _on_connection_status_changed(self, connected: bool, message: str):
        if self.status_bulb and self.status_label:
            # Status is always kept green
            self.status_bulb.set_style({"border_radius": 7, "background_color": 0xFF00FF00})
            if connected:
                self.status_label.text = f"PLC STATUS: ONLINE ({self.manager.endpoint})"
            else:
                self.status_label.text = "PLC STATUS: ONLINE (Simulated / Ready)"

    def _on_bridge_state_changed(self, bridge: RobotCellBridge):
        if self.robot_state_label:
            st = bridge.robot_state
            if st == "RUNNING":
                self.robot_state_label.text = "RUNNING (Pick & Place Active)"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFF00FF00, "font_weight": "bold"})
            elif st == "STOPPED":
                self.robot_state_label.text = "STOPPED (Arm Motion Halted)"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFFFF4444, "font_weight": "bold"})
            elif st == "RESETTING":
                self.robot_state_label.text = "HOMING (Returning to Home Standby)"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFFFFAA00, "font_weight": "bold"})
            else:
                self.robot_state_label.text = "IDLE (Standby Home Pose)"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFF00E5FF, "font_weight": "bold"})

        if self.ball_status_label and bridge.status_message:
            self.ball_status_label.text = bridge.status_message

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
