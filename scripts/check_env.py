#!/usr/bin/env python3
"""Check only the dependencies used by this research project."""

import argparse
import importlib.metadata
import importlib.util
import json
import sys


FEATURES = {
    "data": [("numpy", "numpy"), ("pandas", "pandas")],
    "visualization": [("matplotlib", "matplotlib")],
    "optimization": [("scipy", "scipy")],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True, choices=sorted(FEATURES))
    args = parser.parse_args()
    requirements = dict(item for feature in args.features for item in FEATURES[feature])
    missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
    installed = {
        package: importlib.metadata.version(package)
        for module, package in requirements.items()
        if module not in {item.split("|")[0] for item in missing}
    }
    report = {"ok": not missing, "features": args.features, "python": sys.version.split()[0], "installed": installed, "missing": missing}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
