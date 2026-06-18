from __future__ import annotations

import argparse
import webbrowser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a browser viewer for the Jetson stream.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    webbrowser.open(f"http://{args.host}:{args.port}/")


if __name__ == "__main__":
    main()
