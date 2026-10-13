"""Allow the engine to run as a Python module."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
