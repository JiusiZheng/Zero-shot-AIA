"""Zero-dependency helpers usable before the torch-dependent `backends`
module is imported (so `--inspect` mode never needs torch installed)."""
import sys
from typing import Any


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)
