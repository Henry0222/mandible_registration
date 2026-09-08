"""PyInstaller entry point: the same executable serves GUI and child viewers."""

from mandible_registration.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
