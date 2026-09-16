# SPDX-License-Identifier: Apache-2.0
"""
UI Window for Siemens PLC OPC UA Bridge in Isaac Sim / Omniverse.
Provides server address input, connection status indicator (green/red bulb),
DB/Byte/Bit offset configuration, and 2 ON/OFF toggle buttons.
"""

from __future__ import annotations
import asyncio
import omni.ui as ui
from .opcua_mgr import PLCManager


class PLCBridgeUI:
    """Main window UI for configuring and monitoring Siemens PLC OPC UA connection."""

    def __init__(self, manager: PLCManager):
        self.manager = manager

        # Window creation (Standard dockable Omniverse window)
        self.window = ui.Window("Siemens PLC OPC UA Bridge", width=380, height=540)

        # UI Element handles
        self.endpoint_field: ui.StringField = None
        self.status_bulb: ui.Rectangle = None
        self.status_label: ui.Label = None
        self.connect_button: ui.Button = None

        self.db_field: ui.IntField = None
        self.byte_field: ui.IntField = None
        self.bit_field: ui.IntField = None
        self.symbol_field: ui.StringField = None
        self.effective_node_label: ui.Label = None

        self.toggle_1_btn: ui.Button = None
        self.toggle_2_btn: ui.Button = None
        self.toggle_1_state: bool = False
        self.toggle_2_state: bool = False

        self._build_ui()
        self.manager.register_status_callback(self._on_status_changed)

    @property
    def timeline(self):
        try:
            import omni.timeline
            return omni.timeline.get_timeline_interface()
        except Exception:
            return None

    def _build_ui(self):
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, style={"margin": 10}):
                    # -------------------------------------------------------------
                    # Header
                    # -------------------------------------------------------------
                    ui.Label(
                        "Siemens PLC OPC UA Bridge",
                        style={"font_size": 18, "color": 0xFF00B4D8},
                        alignment=ui.Alignment.CENTER,
                    )
                    ui.Line(style={"color": 0x44FFFFFF})

                    # -------------------------------------------------------------
                    # Section 1: Connection & Status Bulb
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("1. Connection Settings", collapsed=False):
                        with ui.VStack(spacing=6):
                            ui.Label("OPC UA Server URL:")
                            self.endpoint_field = ui.StringField()
                            self.endpoint_field.model.set_value(self.manager.endpoint)

                            # Status row with Green/Red bulb
                            with ui.HStack(height=24, spacing=8):
                                ui.Label("Connection Status:", width=130)
                                # Rounded rectangle acting as glowing bulb
                                self.status_bulb = ui.Rectangle(
                                    width=16,
                                    height=16,
                                    style={"border_radius": 8, "background_color": 0xFFFF2222},  # Initial RED
                                )
                                self.status_label = ui.Label(
                                    "Disconnected",
                                    style={"color": 0xFFFF4444, "font_size": 13},
                                )

                            with ui.HStack(spacing=6):
                                self.connect_button = ui.Button(
                                    "Connect",
                                    clicked_fn=self._on_connect_clicked,
                                    style={"background_color": 0xFF2A9D8F},
                                )
                                ui.Button(
                                    "Disconnect",
                                    clicked_fn=self._on_disconnect_clicked,
                                    style={"background_color": 0xFFE76F51},
                                )

                    # -------------------------------------------------------------
                    # Section 2: Siemens S7 Data Block (DB) Configuration
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("2. Siemens DB Tag Addressing", collapsed=False):
                        with ui.VStack(spacing=6):
                            ui.Label("Configure Start / Control Bit in PLC:")

                            with ui.HStack():
                                ui.Label("DB Number:", width=100)
                                self.db_field = ui.IntField()
                                self.db_field.model.set_value(self.manager.db_number)
                                self.db_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Byte Offset:", width=100)
                                self.byte_field = ui.IntField()
                                self.byte_field.model.set_value(self.manager.byte_offset)
                                self.byte_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Bit Offset:", width=100)
                                self.bit_field = ui.IntField()
                                self.bit_field.model.set_value(self.manager.bit_offset)
                                self.bit_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Custom Tag/Symbol:", width=120)
                                self.symbol_field = ui.StringField()
                                self.symbol_field.model.set_value("")
                                self.symbol_field.model.add_value_changed_fn(self._update_node_display)

                            ui.Label("Effective OPC UA NodeId:", style={"color": 0xFFAAAAAA})
                            self.effective_node_label = ui.Label(
                                self.manager.get_effective_node_id(),
                                style={"color": 0xFFE9C46A, "font_size": 11},
                            )

                    # -------------------------------------------------------------
                    # Section 3: 2 Toggle Buttons (ON / OFF)
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("3. Control Toggles", collapsed=False):
                        with ui.VStack(spacing=8):
                            ui.Label("Manual Testing & Simulation Triggers:")

                            # Toggle 1
                            with ui.HStack(height=32):
                                ui.Label("Toggle 1 (PLC Start Bit):", width=160)
                                self.toggle_1_btn = ui.Button(
                                    "OFF",
                                    clicked_fn=self._on_toggle_1_clicked,
                                    style={"background_color": 0xFF555555, "font_size": 14},
                                )

                            # Toggle 2
                            with ui.HStack(height=32):
                                ui.Label("Toggle 2 (Auxiliary Bit):", width=160)
                                self.toggle_2_btn = ui.Button(
                                    "OFF",
                                    clicked_fn=self._on_toggle_2_clicked,
                                    style={"background_color": 0xFF555555, "font_size": 14},
                                )

                    # -------------------------------------------------------------
                    # Section 4: Timeline Synchronization Info
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("4. Play/Pause Activity", collapsed=False):
                        with ui.VStack(spacing=6):
                            ui.Label(
                                "When PLC Start bit turns ON (True):",
                                style={"font_size": 11, "color": 0xFF00B4D8},
                            )
                            ui.Label(
                                "-> Simulation PLAY starts automatically (PhysX activates).",
                                style={"font_size": 10},
                            )
                            ui.Label(
                                "When PLC Start bit turns OFF (False):",
                                style={"font_size": 11, "color": 0xFFE76F51},
                            )
                            ui.Label(
                                "-> Simulation PAUSES automatically.",
                                style={"font_size": 10},
                            )

                            with ui.HStack():
                                ui.Button(
                                    "Force Play",
                                    clicked_fn=lambda: self.timeline.play(),
                                    style={"background_color": 0xFF2A9D8F},
                                )
                                ui.Button(
                                    "Force Pause",
                                    clicked_fn=lambda: self.timeline.pause(),
                                    style={"background_color": 0xFFE9C46A},
                                )
                                ui.Button(
                                    "Force Stop",
                                    clicked_fn=lambda: self.timeline.stop(),
                                    style={"background_color": 0xFFE76F51},
                                )

    def _update_node_display(self, model=None):
        """Update node id preview and manager variables."""
        if self.db_field:
            self.manager.db_number = self.db_field.model.get_value_as_int()
        if self.byte_field:
            self.manager.byte_offset = self.byte_field.model.get_value_as_int()
        if self.bit_field:
            self.manager.bit_offset = self.bit_field.model.get_value_as_int()
        if self.symbol_field:
            self.manager.custom_node_id = self.symbol_field.model.get_value_as_string()

        if self.effective_node_label:
            self.effective_node_label.text = self.manager.get_effective_node_id()

    def _on_connect_clicked(self):
        url = self.endpoint_field.model.get_value_as_string()
        self._update_node_display()
        asyncio.ensure_future(self.manager.connect(url))

    def _on_disconnect_clicked(self):
        asyncio.ensure_future(self.manager.disconnect())

    def _on_status_changed(self, connected: bool, message: str):
        """Called when OPC UA connection status updates."""
        if self.status_bulb and self.status_label:
            if connected:
                # Green Bulb!
                self.status_bulb.set_style({"border_radius": 8, "background_color": 0xFF00FF00})
                self.status_label.text = "Connected (Online)"
                self.status_label.set_style({"color": 0xFF00FF00, "font_size": 13})
            else:
                # Red Bulb!
                self.status_bulb.set_style({"border_radius": 8, "background_color": 0xFFFF2222})
                self.status_label.text = message
                self.status_label.set_style({"color": 0xFFFF4444, "font_size": 13})

    def _on_toggle_1_clicked(self):
        self.toggle_1_state = not self.toggle_1_state
        if self.toggle_1_state:
            self.toggle_1_btn.text = "ON"
            self.toggle_1_btn.set_style({"background_color": 0xFF00CC44, "font_size": 14})
            if not self.timeline.is_playing():
                self.timeline.play()
        else:
            self.toggle_1_btn.text = "OFF"
            self.toggle_1_btn.set_style({"background_color": 0xFF555555, "font_size": 14})
            if self.timeline.is_playing():
                self.timeline.pause()

        asyncio.ensure_future(self.manager.write_toggle_bit(1, self.toggle_1_state))

    def _on_toggle_2_clicked(self):
        self.toggle_2_state = not self.toggle_2_state
        if self.toggle_2_state:
            self.toggle_2_btn.text = "ON"
            self.toggle_2_btn.set_style({"background_color": 0xFF0088FF, "font_size": 14})
        else:
            self.toggle_2_btn.text = "OFF"
            self.toggle_2_btn.set_style({"background_color": 0xFF555555, "font_size": 14})

        asyncio.ensure_future(self.manager.write_toggle_bit(2, self.toggle_2_state))

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
