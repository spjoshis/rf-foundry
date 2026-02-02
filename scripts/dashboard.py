#!/usr/bin/env python3
"""
Launch the TradeBox-RL Streamlit dashboard.

This is a convenience CLI wrapper around streamlit run.

Usage:
    python scripts/dashboard.py [--db-path PATH] [--port PORT]

    Or directly with Streamlit:
    streamlit run src/tradebox/dashboard/app.py

Examples:
    # Launch with default settings (data/metrics.db, port 8501)
    python scripts/dashboard.py

    # Custom database path
    python scripts/dashboard.py --db-path /path/to/metrics.db

    # Custom port
    python scripts/dashboard.py --port 8080

    # Both custom
    python scripts/dashboard.py --db-path data/live_metrics.db --port 8080
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Launch Streamlit dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch TradeBox-RL Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="data/metrics.db",
        help="Path to metrics database (default: data/metrics.db)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run dashboard on (default: 8501)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Auto-open browser (default: True)",
    )

    parser.add_argument(
        "--app",
        type=str,
        default="app",
        help="app / report",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser",
    )

    args = parser.parse_args()

    # Locate app.py
    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "tradebox" / "dashboard" / f"{args.app}.py"

    if not app_path.exists():
        print(f"❌ Error: Dashboard app not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Check if streamlit is installed
    try:
        subprocess.run(
            ["streamlit", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Streamlit not installed", file=sys.stderr)
        print("Install with: poetry add streamlit plotly", file=sys.stderr)
        sys.exit(1)

    # Build streamlit command
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    # Browser settings
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    else:
        cmd.extend(["--server.headless", "false"])

    # Theme
    cmd.extend([
        "--theme.primaryColor", "#1f77b4",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
        "--theme.textColor", "#262730",
    ])

    # Print launch info
    print("=" * 60)
    print("🚀 Launching TradeBox-RL Dashboard")
    print("=" * 60)
    print(f"📊 App: {app_path.name}")
    print(f"💾 Database: {args.db_path}")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print(f"🔄 Auto-refresh: Enabled (configurable in sidebar)")
    print("=" * 60)
    print("\n✨ Dashboard starting...\n")

    # Launch streamlit
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
