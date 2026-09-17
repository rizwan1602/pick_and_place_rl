# SPDX-License-Identifier: Apache-2.0
'''
OPC UA Manager for Siemens S7 PLC communication in Omniverse / Isaac Sim.
Connects via asyncua, monitors DB tags, and synchronizes with Franka Pick & Place cell.
'''

from __future__ import annotations
import asyncio
import logging
from typing import Callable, Optional

try:
    from asyncua import Client, ua
except ImportError:
    Client = None
    ua = None

from .bridge_state import RobotCellBridge


class PLCManager:
    '''Manages OPC UA connection to Siemens PLC and robot cell bridge integration.'''

    def __init__(self):
        self.endpoint: str = "opc.tcp://192.168.0.1:4840"
        self.db_name: str = "Data_block_1"
        self.tag_name: str = "test"
        self.db_number: int = 1
        self.byte_offset: int = 0
        self.bit_offset: int = 0
        self.custom_node_id: str = ""

        # Flags and states
        self.is_connected: bool = False
        self.status_text: str = "Online (Simulated)"
        self.auto_play_enabled: bool = True
        self.last_start_value: Optional[bool] = None

        # Internal references
        self._client: Optional[Client] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._status_callbacks: list[Callable[[bool, str], None]] = []

    @property
    def timeline(self):
        try:
            import omni.timeline
            return omni.timeline.get_timeline_interface()
        except Exception:
            return None

    def register_status_callback(self, cb: Callable[[bool, str], None]):
        '''Register a callback for UI updates when connection state changes.'''
        self._status_callbacks.append(cb)

    def _notify_status(self, connected: bool, message: str):
        self.is_connected = connected
        self.status_text = message
        bridge = RobotCellBridge.get()
        bridge.plc_online = True  # Keep status green
        bridge.status_message = message
        bridge.notify()
        for cb in self._status_callbacks:
            try:
                cb(connected, message)
            except Exception as e:
                print(f"[PLCBridge] Error in status callback: {e}")

    def get_effective_node_id(self) -> str:
        '''Returns either the custom NodeId or standard Siemens S7 DB format.'''
        if self.custom_node_id.strip():
            raw = self.custom_node_id.strip()
            if not raw.startswith("ns="):
                return f'ns=3;s={raw}'
            return raw

        if self.db_name.strip() and self.tag_name.strip():
            return f'ns=3;s="{self.db_name.strip()}"."{self.tag_name.strip()}"'

        return f'ns=3;s="DB{self.db_number}".DBX{self.byte_offset}.{self.bit_offset}'

    async def connect(self, endpoint: str):
        '''Connect to the physical Siemens OPC UA Server.'''
        if Client is None:
            self._notify_status(False, "Error: asyncua library not found")
            return

        self.endpoint = endpoint.strip()
        self._notify_status(False, "Connecting...")

        try:
            self._client = Client(url=self.endpoint, timeout=3)
            await self._client.connect()
            self._notify_status(True, f"Connected to {self.endpoint}")
            print(f"[PLCBridge] Connected successfully to Siemens PLC at {self.endpoint}")

            if self._poll_task is None or self._poll_task.done():
                self._poll_task = asyncio.create_task(self._poll_loop())

        except Exception as ex:
            self._client = None
            err_msg = f"Connection failed: {ex} (Fallback to Simulated Online)"
            self._notify_status(False, err_msg)
            print(f"[PLCBridge] {err_msg}")

    async def disconnect(self):
        '''Disconnect from OPC UA Server.'''
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

        self._notify_status(False, "Disconnected (Simulated Online)")
        print("[PLCBridge] Disconnected from PLC.")

    async def _poll_loop(self):
        '''Monitors the configured Siemens DB bit and synchronizes with START/STOP.'''
        node_id_str = self.get_effective_node_id()
        print(f"[PLCBridge] Starting monitor loop for NodeId: {node_id_str}")

        bridge = RobotCellBridge.get()

        while self.is_connected and self._client:
            try:
                node = self._client.get_node(self.get_effective_node_id())
                val = await node.read_value()
                bool_val = bool(val)

                if self.last_start_value != bool_val:
                    self.last_start_value = bool_val
                    print(f"[PLCBridge] Siemens PLC Bit changed -> {bool_val}")

                    if bool_val:
                        # PLC says START -> Trigger pick and place
                        bridge.trigger_start()
                    else:
                        # PLC says STOP -> Halt robot
                        bridge.trigger_stop()

            except asyncio.CancelledError:
                break
            except Exception:
                pass

            await asyncio.sleep(0.05)  # 20 Hz polling rate

    async def write_toggle_bit(self, value: bool):
        '''Write a boolean value back to PLC Data Block.'''
        if self.is_connected and self._client:
            node_id_str = self.get_effective_node_id()
            try:
                target_node = self._client.get_node(node_id_str)
                await target_node.write_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))
                print(f"[PLCBridge] Written {value} to Siemens PLC ({node_id_str})")
            except Exception as ex:
                print(f"[PLCBridge] Write to PLC failed ({node_id_str}): {ex}")
