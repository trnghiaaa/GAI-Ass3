"""Promote a validated checkpoint without discarding its predecessor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from stable_baselines3 import DQN

from arena.cooldown import load_dqn
from arena.environment import ArenaEnv, ENVIRONMENT_SCHEMA_VERSION, OBSERVATION_NAMES
from arena.settings import metadata_path, model_path
from arena.train import transfer_prefix_policy


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_metadata(
    metadata: dict[str, Any],
    destination: Path,
    source: Path,
    holdout: Path,
    mode: str,
) -> None:
    metrics = _load_json(holdout)["aggregate"]
    result = copy.deepcopy(metadata)
    result.update(
        {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "observation_names": list(OBSERVATION_NAMES),
            "observation_size": len(OBSERVATION_NAMES),
            "benchmark": metrics,
            "promoted_from": str(source),
            "promoted_from_sha256": _sha256(source),
            "holdout_evaluation": str(holdout),
            "promotion_mode": mode,
        }
    )
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def promote_checkpoint(source: Path, style: str, metadata: Path, holdout: Path) -> None:
    source_metadata = _load_json(metadata)
    if source_metadata["control_style"] != style:
        raise ValueError("checkpoint control style does not match destination")
    if source_metadata["observation_names"] != list(OBSERVATION_NAMES):
        raise ValueError("checkpoint does not match the live observation schema")
    destination = model_path(style)
    shutil.copy2(source, destination)
    _write_metadata(source_metadata, destination, source, holdout, "validated_checkpoint")


def promote_prefix(source: Path, style: str, metadata: Path, holdout: Path) -> None:
    source_metadata = _load_json(metadata)
    if source_metadata["control_style"] != style:
        raise ValueError("baseline control style does not match destination")
    source_model = load_dqn(str(source), device="cpu")
    old_size = int(source_model.observation_space.shape[0])
    if source_metadata["observation_names"] != list(OBSERVATION_NAMES[:old_size]):
        raise ValueError("baseline is not an exact observation-prefix match")
    env = ArenaEnv(control_style=style)
    try:
        destination_model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": list(source_metadata["network"])},
            device="cpu",
            verbose=0,
        )
        transfer_prefix_policy(source_model, destination_model)
        destination_model.num_timesteps = source_model.num_timesteps
        destination = model_path(style)
        destination_model.save(str(destination))
    finally:
        env.close()
    # The transferred Q-values are exactly those logged in this original run;
    # keeping that run name lets evidence tooling trace their real provenance.
    source_metadata["run_name"] = "dqn_direct"
    source_metadata["initialization_model"] = str(source)
    _write_metadata(source_metadata, destination, source, holdout, "frozen_prefix_compatibility")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint", "prefix"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--control-style", choices=("rotation", "direct"), required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    if not args.holdout.is_file():
        raise FileNotFoundError(args.holdout)
    if args.mode == "checkpoint":
        promote_checkpoint(args.source, args.control_style, args.source_metadata, args.holdout)
    else:
        promote_prefix(args.source, args.control_style, args.source_metadata, args.holdout)
    print(f"Promoted {args.control_style} model to {model_path(args.control_style)}")


if __name__ == "__main__":
    main()
