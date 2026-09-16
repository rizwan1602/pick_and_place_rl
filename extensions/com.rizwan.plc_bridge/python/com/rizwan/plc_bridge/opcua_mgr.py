# SPDX-License-Identifier: Apache-2.0
"""
OPC UA Manager for Siemens S7 PLC communication in Omniverse / Isaac Sim.
Connects via asyncua, monitors DB tags, and synchronizes simulation Play/Pause.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Callable, Optional

import omni.timeline
import omni.kit.app

try:
    from asyncua import Client, ua
except ImportError:
    Client = None
    ua = None


class PLCManager:
    """Manages OPC UA connection to Siemens PLC and timeline integration."""

    def __init__(self):
        self.endpoint: str = "opc.tcp://192.168.0.1:4840"
        self.db_number: int = 100
        self.byte_offset: int = 0
        self.bit_offset: int = 0
        self.custom_node_id: str = ""

        # Flags and states
        self.is_connected: bool = False
        self.status_text: str = "Disconnected"
        self.auto_play_enabled: bool = True
        self.toggle_1_active: bool = False
        self.toggle_2_active: bool = False
        self.last_start_value: Optional[bool] = None

        # Internal references
        self._client: Optional[Client] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._status_callbacks: list[Callable[[bool, str], None]] = []

    @property
    def timeline(self):
        return omni.timeline.get_timeline_interface()

    def register_status_callback(self, cb: Callable[[bool, str], None]):
        """Register a callback for UI updates when connection state changes."""
        self._status_callbacks.append(cb)

    def _notify_status(self, connected: bool, message: str):
        self.is_connected = connected
        self.status_text = message
        for cb in self._status_callbacks:
            try:
                cb(connected, message)
            except Exception as e:
                print(f"[PLCBridge] Error in status callback: {e}")

    def get_effective_node_id(self) -> str:
        """Returns either the custom NodeId or standard Siemens S7 DB format."""
        if self.custom_node_id.strip():
            return self.custom_node_id.strip()
        # Common Siemens S7 OPC UA format for optimized DB:
        # e.g., ns=3;s="DB100"."bStart" or raw address DBX0.0
        return f'ns=3;s="DB{self.db_number}".DBX{self.byte_offset}.{self.bit_offset}'

    async def connect(self, endpoint: str):
        """Connect to the OPC UA Server."""
        if Client is None:
            self._notify_status(False, "Error: asyncua library not found")
            return

        self.endpoint = endpoint.strip()
        self._notify_status(False, "Connecting...")

        try:
            self._client = Client(url=self.endpoint, timeout=4)
            await self._client.connect()
            self._notify_status(True, f"Connected to {self.endpoint}")
            print(f"[PLCBridge] Connected successfully to {self.endpoint}")

            # Start background monitoring loop
            if self._poll_task is None or self._poll_task.done():
                self._poll_task = asyncio.create_task(self._poll_loop())

        except Exception as ex:
            self._client = None
            err_msg = f"Connection failed: {ex}"
            self._notify_status(False, err_msg)
            print(f"[PLCBridge] {err_msg}")

    async def disconnect(self):
        """Disconnect from OPC UA Server."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

        self._notify_status(False, "Disconnected")
        print("[PLCBridge] Disconnected from PLC.")

    async def _poll_loop(self):
        """Monitors the configured Siemens DB bit and synchronizes with Play button."""
        node_id_str = self.get_effective_node_id()
        print(f"[PLCBridge] Starting monitor loop for NodeId: {node_id_str}")

        while self.is_connected and self._client:
            try:
                node = self._client.get_node(self.get_effective_node_id())
                val = await node.read_value()
                bool_val = bool(val)

                # If PLC state changed
                if self.last_start_value != bool_val:
                    self.last_start_value = bool_val
                    print(f"[PLCBridge] PLC Start Bit changed -> {bool_val}")

                    if self.auto_play_enabled and self.timeline:
                        if bool_val:
                            # PLC says START -> Isaac Sim Timeline PLAY
                            if not self.timeline.is_playing():
                                print("[PLCBridge] Triggering Isaac Sim PLAY from PLC signal!")
                                self.timeline.play()
                        else:
                            # PLC says STOP -> Isaac Sim Timeline PAUSE
                            if self.timeline.is_playing():
                                print("[PLCBridge] Pausing Isaac Sim from PLC signal!")
                                self.timeline.pause()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Could be node not found or connection lost
                pass

            await asyncio.sleep(0.05)  # 20 Hz polling rate

    async def write_toggle_bit(self, toggle_index: int, value: bool):
        """Write a boolean toggle value back to PLC or simulation."""
        if toggle_index == 1:
            self.toggle_1_active = value
        elif toggle_index == 2:
            self.toggle_2_active = value

        print(f"[PLCBridge] Toggle {toggle_index} set to: {value}")

        # If connected, attempt to write to target node if available
        if self.is_connected and self._client:
            try:
                target_node = self._client.get_node(self.get_effective_node_id())
                await target_node.write_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))
                print(f"[PLCBridge] Written {value} to PLC successfully")
            except Exception as ex:
                print(f"[PLCBridge] Write to PLC failed (tag may be read-only): {ex}")
