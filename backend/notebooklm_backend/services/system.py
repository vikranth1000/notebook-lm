from __future__ import annotations

import platform
from datetime import datetime


def system_probe() -> str:
    """
    Returns a string summarising host platform and time.
    This is a lightweight placeholder for deeper telemetry that should remain on-device.
    """
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    timestamp = datetime.utcnow().isoformat()
    return f"{system} {release} ({machine}) @ {timestamp}Z"

