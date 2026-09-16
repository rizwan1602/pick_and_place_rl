# SPDX-License-Identifier: Apache-2.0
"""
UI Window for Siemens PLC OPC UA Bridge in Isaac Sim / Omniverse.
Provides server address input, live status indicator, DB tag configuration,
and 2 dedicated TRUE / FALSE buttons with auto-connect.
"""

from __future__ import annotations
import asyncio
import omni.ui as ui
from .opcua_mgr import PLCManager


class PLCBridgeUI:
    """Main window UI for configuring and monitoring Siemens PLC OPC UA connection."""

    def __init__(self, manager: PLCManager):
        self.manager = manager

        # Window creation
        self.window = ui.Window("Siemens PLC OPC UA Bridge", width=380, height=480)

        # UI Element handles
        self.endpoint_field: ui.StringField = None
        self.status_bulb: ui.Rectangle = None
        self.status_label: ui.Label = None
        self.connect_button: ui.Button = None

        self.db_name_field: ui.StringField = None
        self.tag_name_field: ui.StringField = None
        self.symbol_field: ui.StringField = None
        self.effective_node_label: ui.Label = None

        self.tag_status_label: ui.Label = None
        self.btn_true: ui.Button = None
        self.btn_false: ui.Button = None

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
                                self.status_bulb = ui.Rectangle(
                                    width=16,
                                    height=16,
                                    style={"border_radius": 8, "background_color": 0xFFFF2222},  # Red
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
                    # Section 2: Siemens S7 Tag Control (Includes TRUE / FALSE Buttons)
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("2. Siemens DB Tag Control", collapsed=False):
                        with ui.VStack(spacing=8):
                            ui.Label("Configure Target Tag in PLC:")

                            with ui.HStack():
                                ui.Label("DB Name:", width=100)
                                self.db_name_field = ui.StringField()
                                self.db_name_field.model.set_value(self.manager.db_name)
                                self.db_name_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Tag Name:", width=100)
                                self.tag_name_field = ui.StringField()
                                self.tag_name_field.model.set_value(self.manager.tag_name)
                                self.tag_name_field.model.add_value_changed_fn(self._update_node_display)

                            with ui.HStack():
                                ui.Label("Custom NodeId:", width=120)
                                self.symbol_field = ui.StringField()
                                self.symbol_field.model.set_value("")
                                self.symbol_field.model.add_value_changed_fn(self._update_node_display)

                            ui.Label("Effective OPC UA NodeId:", style={"color": 0xFFAAAAAA})
                            self.effective_node_label = ui.Label(
                                self.manager.get_effective_node_id(),
                                style={"color": 0xFFE9C46A, "font_size": 12},
                            )

                            ui.Line(style={"color": 0x22FFFFFF})

                            # 2 Dedicated Buttons: TRUE and FALSE
                            ui.Label("Write Bit to PLC & Control Simulation:", style={"font_size": 12, "color": 0xFF00B4D8})
                            with ui.HStack(height=38, spacing=10):
                                self.btn_true = ui.Button(
                                    "SET TRUE  (1)",
                                    clicked_fn=self._on_set_true_clicked,
                                    style={"background_color": 0xFF00AA44, "font_size": 14, "border_radius": 4},
                                )
                                self.btn_false = ui.Button(
                                    "SET FALSE (0)",
                                    clicked_fn=self._on_set_false_clicked,
                                    style={"background_color": 0xFFCC3333, "font_size": 14, "border_radius": 4},
                                )

                            self.tag_status_label = ui.Label(
                                "Current Tag State: Ready",
                                style={"color": 0xFFDDDDDD, "font_size": 12},
                            )

                    # -------------------------------------------------------------
                    # Section 3: Timeline Synchronization Info
                    # -------------------------------------------------------------
                    with ui.CollapsableFrame("3. Play/Pause Activity", collapsed=True):
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

    def _update_node_display(self, model=None):
        """Update node id preview and manager variables."""
        if hasattr(self, "db_name_field") and self.db_name_field:
            self.manager.db_name = self.db_name_field.model.get_value_as_string()
        if hasattr(self, "tag_name_field") and self.tag_name_field:
            self.manager.tag_name = self.tag_name_field.model.get_value_as_string()
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
                if self.tag_status_label:
                    self.tag_status_label.text = "Connected to PLC! Ready to send."
                    self.tag_status_label.set_style({"color": 0xFF00FF00, "font_size": 12})
            else:
                # Red Bulb!
                self.status_bulb.set_style({"border_radius": 8, "background_color": 0xFFFF2222})
                self.status_label.text = message
                self.status_label.set_style({"color": 0xFFFF4444, "font_size": 13})
                if self.tag_status_label:
                    self.tag_status_label.text = f"Status: {message}"
                    self.tag_status_label.set_style({"color": 0xFFFF4444, "font_size": 12})

    def _on_set_true_clicked(self):
        """Write True to PLC and start physics simulation."""
        self._update_node_display()
        if not self.manager.is_connected:
            url = self.endpoint_field.model.get_value_as_string() if self.endpoint_field else self.manager.endpoint
            if self.tag_status_label:
                self.tag_status_label.text = "Connecting to PLC and writing TRUE..."
                self.tag_status_label.set_style({"color": 0xFFE9C46A, "font_size": 12})
            asyncio.ensure_future(self._connect_and_write(url, True))
            return

        asyncio.ensure_future(self.manager.write_toggle_bit(1, True))
        if self.tag_status_label:
            self.tag_status_label.text = "Sent TRUE (1) to PLC -> Simulation Playing"
            self.tag_status_label.set_style({"color": 0xFF00FF00, "font_size": 12})
        if self.timeline and not self.timeline.is_playing():
            self.timeline.play()

    def _on_set_false_clicked(self):
        """Write False to PLC and pause physics simulation."""
        self._update_node_display()
        if not self.manager.is_connected:
            url = self.endpoint_field.model.get_value_as_string() if self.endpoint_field else self.manager.endpoint
            if self.tag_status_label:
                self.tag_status_label.text = "Connecting to PLC and writing FALSE..."
                self.tag_status_label.set_style({"color": 0xFFE9C46A, "font_size": 12})
            asyncio.ensure_future(self._connect_and_write(url, False))
            return

        asyncio.ensure_future(self.manager.write_toggle_bit(1, False))
        if self.tag_status_label:
            self.tag_status_label.text = "Sent FALSE (0) to PLC -> Simulation Paused"
            self.tag_status_label.set_style({"color": 0xFFFF4444, "font_size": 12})
        if self.timeline and self.timeline.is_playing():
            self.timeline.pause()

    async def _connect_and_write(self, url: str, val: bool):
        """Auto-connects if disconnected, then writes value."""
        await self.manager.connect(url)
        if self.manager.is_connected:
            await self.manager.write_toggle_bit(1, val)
            if self.tag_status_label:
                state_txt = "TRUE (1)" if val else "FALSE (0)"
                self.tag_status_label.text = f"Connected! Wrote {state_txt} to PLC"
                self.tag_status_label.set_style({"color": 0xFF00FF00 if val else 0xFFFF4444, "font_size": 12})
            if val:
                if self.timeline and not self.timeline.is_playing():
                    self.timeline.play()
            else:
                if self.timeline and self.timeline.is_playing():
                    self.timeline.pause()

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
