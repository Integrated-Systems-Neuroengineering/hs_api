#!/usr/bin/env python3
from hs_api.api import *
try:
    from hs_bridge.FPGA_Execution.fpga_controller import ACKError, ACK_STATUS_OK, ACK_STATUS_BACKPRESSURE
except ImportError:
    pass  # hs_bridge not installed (software-sim-only environment)
