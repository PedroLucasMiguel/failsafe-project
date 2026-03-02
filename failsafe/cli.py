"""
CLI entry point — `failsafe`.

Runs the interactive setup wizard, then the main menu.
All configuration is collected interactively; nothing is read from flags.
"""

from __future__ import annotations


def main() -> None:
    """Failsafe CLI entry point."""
    from failsafe.ui.setup import run_setup
    from failsafe.ui.menu import run_menu

    config = run_setup()
    run_menu(config)


if __name__ == "__main__":
    main()
