# SPDX-License-Identifier: Apache-2.0
"""
Main entry point for siemens_plc extension.
"""

from __future__ import annotations
import asyncio
import omni.ext
from .opcua_mgr import PLCManager
from .ui import PLCBridgeUI


class Extension(omni.ext.IExt):
    """Main extension class instantiated by Omniverse Kit."""

    def on_startup(self, ext_id: str):
        print(f"[Siemens PLC] Extension startup: {ext_id}")
        self.manager = PLCManager()
        try:
            self.ui = PLCBridgeUI(self.manager)
            print("[Siemens PLC] UI created successfully")
        except Exception as e:
            print(f"[Siemens PLC] ERROR during UI initialization: {e}")
            import traceback
            traceback.print_exc()

    def on_shutdown(self):
        print("[Siemens PLC] Extension shutdown")
        if hasattr(self, "manager") and self.manager and self.manager.is_connected:
            asyncio.ensure_future(self.manager.disconnect())

        if hasattr(self, "ui") and self.ui:
            self.ui.destroy()
            self.ui = None

        self.manager = None
