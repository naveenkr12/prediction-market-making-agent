from __future__ import annotations

import re
from datetime import datetime, timezone


def iso_to_unix(iso_ts: str) -> int | None:
    if not iso_ts:
        return None
    try:
        return int(datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def unix_to_iso(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def safe_filename(ticker: str) -> str:
    """Kalshi tickers can contain characters that are awkward in filenames on some
    filesystems; keep it readable but safe."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
