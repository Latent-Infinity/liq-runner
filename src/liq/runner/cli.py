"""CLI for pipeline spec validation and dry-run output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from liq.runner.pipeline_spec import PipelineSpec, compute_config_hash, dry_run_steps


def _load_spec(path: Path) -> PipelineSpec:
    payload = json.loads(path.read_text())
    spec = PipelineSpec.from_dict(payload)
    spec.validate()
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="liq-runner pipeline tools")
    parser.add_argument("--spec", required=True, help="Path to pipeline spec JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print dry-run stage plan")
    parser.add_argument("--print-hash", action="store_true", help="Print config hash")

    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")

    spec = _load_spec(spec_path)
    output: dict[str, Any] = {"name": spec.name}

    if args.dry_run:
        output["dry_run"] = dry_run_steps(spec)
    if args.print_hash:
        output["config_hash"] = compute_config_hash(spec)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
