#!/usr/bin/env python3
"""Compatibility entry point for the strict current-source execution publisher.

Use publish_execution.py --input DIR [--input RERUN] --dataset UNIQUE --label h100|b200.
Incomplete datasets are never published.
"""
from publish_execution import main
if __name__ == '__main__':
    raise SystemExit(main())
