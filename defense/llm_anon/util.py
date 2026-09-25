"""Zero-dependency helpers usable by every sibling module without creating
an import cycle (data.py and model.py both need to log via eprint, and
driver.py imports both of them)."""
import sys
from typing import Any


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)
