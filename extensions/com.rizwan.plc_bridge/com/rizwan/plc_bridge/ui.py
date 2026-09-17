# SPDX-License-Identifier: Apache-2.0
"""
UI Window for Siemens PLC OPC UA Bridge in Isaac Sim / Omniverse.
Industrial High-Contrast Panel with START / STOP / RESET and Ball Controls.
Automatically docks into the bottom-right Property panel.
"""

from __future__ import annotations
import asyncio
import omni.ui as ui
from .opcua_mgr import PLCManager
from .bridge_state import RobotCellBridge


class PLCBridgeUI:
    """Main window UI for configuring Siemens PLC OPC UA and Franka Pick-and-Place cell."""

    def __init__(self, manager: PLCManager):
        self.manager = manager
        self.bridge = RobotCellBridge.get()

        # Compact Window docked in bottom right (Property pane)
        self.window = ui.Window(
            "Siemens PLC OPC UA Bridge",
            width=350,
            height=400,
            dock_preference=ui.DockPreference.RIGHT_BOTTOM,
        )

        # UI Element handles
        self.status_bulb: ui.Rectangle = None
        self.status_label: ui.Label = None
        self.robot_state_label: ui.Label = None

        self.btn_start: ui.Button = None
        self.btn_stop: ui.Button = None
        self.btn_reset: ui.Button = None
        self.btn_ball: ui.Button = None

        self.ball_status_label: ui.Label = None

        self.endpoint_field: ui.StringField = None
        self.db_name_field: ui.StringField = None
        self.tag_name_field: ui.StringField = None
        self.effective_node_label: ui.Label = None

        self._build_ui()
        self.manager.register_status_callback(self._on_connection_status_changed)
        self.bridge.register_listener(self._on_bridge_state_changed)

        # Auto-dock into bottom-right Property tab panel
        asyncio.ensure_future(self._dock_window())

    async def _dock_window(self):
        """Automatically docks window into the bottom-right Property pane as a tab."""
        import omni.kit.app
        for _ in range(15):
            if ui.Workspace.get_window("Siemens PLC OPC UA Bridge"):
                break
            await omni.kit.app.get_app().next_update_async()

        custom_window = ui.Workspace.get_window("Siemens PLC OPC UA Bridge")
        property_window = ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
            custom_window.focus()
            print("[PLCBridge] Successfully docked into Property tab pane!")

    def _build_ui(self):
        # High contrast, large typography button styles
        style_start = {
            "Button": {
                "background_color": 0xFF15803D,  # Deep Emerald Green
                "border_color": 0xFF22C55E,      # Bright Green Border
                "border_width": 2,
                "border_radius": 6,
                "color": 0xFFFFFFFF,             # Pure White Text
                "font_size": 15,
                "font_weight": "bold",
            },
            "Button:hovered": {
                "background_color": 0xFF16A34A,
                "border_color": 0xFF4ADE80,
            },
            "Button:pressed": {
                "background_color": 0xFF14532D,
            }
        }

        style_stop = {
            "Button": {
                "background_color": 0xFFB91C1C,  # Deep Crimson Red
                "border_color": 0xFFEF4444,      # Bright Red Border
                "border_width": 2,
                "border_radius": 6,
                "color": 0xFFFFFFFF,             # Pure White Text
                "font_size": 15,
                "font_weight": "bold",
            },
            "Button:hovered": {
                "background_color": 0xFFDC2626,
                "border_color": 0xFFF87171,
            },
            "Button:pressed": {
                "background_color": 0xFF7F1D1D,
            }
        }

        style_reset = {
            "Button": {
                "background_color": 0xFFC2410C,  # Deep Industrial Amber / Orange
                "border_color": 0xFFFB923C,      # Amber Border
                "border_width": 2,
                "border_radius": 6,
                "color": 0xFFFFFFFF,             # Pure White Text
                "font_size": 15,
                "font_weight": "bold",
            },
            "Button:hovered": {
                "background_color": 0xFFEA580C,
                "border_color": 0xFFFDBA74,
            },
            "Button:pressed": {
                "background_color": 0xFF7C2D12,
            }
        }

        style_ball = {
            "Button": {
                "background_color": 0xFF3730A3,  # Deep Indigo
                "border_color": 0xFF6366F1,      # Bright Indigo Border
                "border_width": 2,
                "border_radius": 6,
                "color": 0xFFFFFFFF,             # Pure White Text
                "font_size": 13,
                "font_weight": "bold",
            },
            "Button:hovered": {
                "background_color": 0xFF4338CA,
                "border_color": 0xFF818CF8,
            },
            "Button:pressed": {
                "background_color": 0xFF312E81,
            }
        }

        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, style={"margin": 8}):
                    # ── Card 1: Top Status Banner ────────────────────────────
                    with ui.ZStack(height=34):
                        ui.Rectangle(style={"background_color": 0x33000000, "border_radius": 5, "border_color": 0x33FFFFFF, "border_width": 1})
                        with ui.HStack(spacing=8, style={"margin": 6}):
                            self.status_bulb = ui.Rectangle(
                                width=12,
                                height=12,
                                style={"border_radius": 6, "background_color": 0xFF00FF00},
                            )
                            self.status_label = ui.Label(
                                "PLC STATUS: ONLINE",
                                style={"color": 0xFF00FF00, "font_size": 13, "font_weight": "bold"},
                            )
                            ui.Spacer()
                            self.robot_state_label = ui.Label(
                                f"[{self.bridge.robot_state}]",
                                style={"font_size": 12, "color": 0xFF00E5FF, "font_weight": "bold"},
                            )

                    # ── Card 2: 3 Big Industrial Control Buttons ─────────────
                    with ui.HStack(height=50, spacing=6):
                        self.btn_start = ui.Button(
                            "START",
                            clicked_fn=self._on_start_clicked,
                            style=style_start,
                        )
                        self.btn_stop = ui.Button(
                            "STOP",
                            clicked_fn=self._on_stop_clicked,
                            style=style_stop,
                        )
                        self.btn_reset = ui.Button(
                            "RESET",
                            clicked_fn=self._on_reset_clicked,
                            style=style_reset,
                        )

                    ui.Label(
                        "RESET resets all: Returns Arm to Home and places ball on table.",
                        style={"font_size": 10, "color": 0xFFAAAAAA},
                    )

                    # ── Card 3: Big Ball Placement Button ────────────────────
                    with ui.HStack(height=44):
                        self.btn_ball = ui.Button(
                            "RANDOMIZE BALL ON TABLE",
                            clicked_fn=self._on_spawn_ball_clicked,
                            style=style_ball,
                        )

                    self.ball_status_label = ui.Label(
                        "Status: Ready. Place ball on table and click START.",
                        style={"font_size": 10, "color": 0xFF00FF88},
                    )

                    ui.Line(style={"color": 0x22FFFFFF})

                    # ── Card 4: Siemens S7 OPC UA Connection (Compact) ───────
                    with ui.CollapsableFrame("Siemens S7 OPC UA Settings", collapsed=True):
                        with ui.VStack(spacing=5, style={"margin": 4}):
                            with ui.HStack():
                                ui.Label("URL:", width=50, style={"font_size": 10})
                                self.endpoint_field = ui.StringField()
                                self.endpoint_field.model.set_value(self.manager.endpoint)

                            with ui.HStack():
                                ui.Label("DB:", width=50, style={"font_size": 10})
                                self.db_name_field = ui.StringField()
                                self.db_name_field.model.set_value(self.manager.db_name)
                                self.db_name_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Tag:", width=50, style={"font_size": 10})
                                self.tag_name_field = ui.StringField()
                                self.tag_name_field.model.set_value(self.manager.tag_name)
                                self.tag_name_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack(height=18):
                                ui.Label("NodeId:", width=50, style={"font_size": 9, "color": 0xFFAAAAAA})
                                self.effective_node_label = ui.Label(
                                    self.manager.get_effective_node_id(),
                                    style={"color": 0xFFE9C46A, "font_size": 10},
                                )

                            with ui.HStack(height=28, spacing=4):
                                ui.Button(
                                    "Connect PLC",
                                    clicked_fn=self._on_connect_clicked,
                                    style={"background_color": 0xFF059669, "font_size": 10},
                                )
                                ui.Button(
                                    "Disconnect",
                                    clicked_fn=self._on_disconnect_clicked,
                                    style={"background_color": 0xFFDC2626, "font_size": 10},
                                )

                            with ui.HStack(height=26, spacing=4):
                                ui.Button(
                                    "Write Bit TRUE (1)",
                                    clicked_fn=lambda: asyncio.ensure_future(self.manager.write_toggle_bit(True)),
                                    style={"background_color": 0xFF047857, "font_size": 9},
                                )
                                ui.Button(
                                    "Write Bit FALSE (0)",
                                    clicked_fn=lambda: asyncio.ensure_future(self.manager.write_toggle_bit(False)),
                                    style={"background_color": 0xFF991B1B, "font_size": 9},
                                )

    def _update_node_display(self, model=None):
        if self.db_name_field:
            self.manager.db_name = self.db_name_field.model.get_value_as_string()
        if self.tag_name_field:
            self.manager.tag_name = self.tag_name_field.model.get_value_as_string()
        if self.effective_node_label:
            self.effective_node_label.text = self.manager.get_effective_node_id()

    def _on_start_clicked(self):
        """User clicked START button in UI."""
        print("[PLC Bridge UI] START button clicked!")
        self.bridge.trigger_start()

    def _on_stop_clicked(self):
        """User clicked STOP button in UI."""
        print("[PLC Bridge UI] STOP button clicked!")
        self.bridge.trigger_stop()

    def _on_reset_clicked(self):
        """User clicked RESET button in UI."""
        print("[PLC Bridge UI] RESET button clicked -> Homing arm and resetting ball to table!")
        self.bridge.trigger_reset()

    def _on_spawn_ball_clicked(self):
        """Randomize ball position on table."""
        print("[PLC Bridge UI] Randomize Ball clicked!")
        self.bridge.trigger_spawn_ball()

    def _on_connect_clicked(self):
        url = self.endpoint_field.model.get_value_as_string()
        self._update_node_display()
        asyncio.ensure_future(self.manager.connect(url))

    def _on_disconnect_clicked(self):
        asyncio.ensure_future(self.manager.disconnect())

    def _on_connection_status_changed(self, connected: bool, message: str):
        if self.status_bulb and self.status_label:
            self.status_bulb.set_style({"border_radius": 6, "background_color": 0xFF00FF00})
            if connected:
                self.status_label.text = f"PLC STATUS: ONLINE ({self.manager.endpoint})"
            else:
                self.status_label.text = "PLC STATUS: ONLINE (Simulated / Ready)"

    def _on_bridge_state_changed(self, bridge: RobotCellBridge):
        if self.robot_state_label:
            st = bridge.robot_state
            if st == "RUNNING":
                self.robot_state_label.text = "[RUNNING]"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFF00FF00, "font_weight": "bold"})
            elif st == "STOPPED":
                self.robot_state_label.text = "[STOPPED]"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFFFF4444, "font_weight": "bold"})
            elif st == "RESETTING":
                self.robot_state_label.text = "[HOMING / RESET]"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFFFFAA00, "font_weight": "bold"})
            else:
                self.robot_state_label.text = "[IDLE / HOME]"
                self.robot_state_label.set_style({"font_size": 12, "color": 0xFF00E5FF, "font_weight": "bold"})

        if self.ball_status_label and bridge.status_message:
            self.ball_status_label.text = bridge.status_message

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
