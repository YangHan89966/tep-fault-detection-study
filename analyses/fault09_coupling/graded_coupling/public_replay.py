"""Portable entry points for the public graded-coupling distribution.

Recalculation and reconstruction run in a separate copy. Archived numerical
results and historical freeze records in the public distribution stay intact.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
from public_identity import sha, verify_frozen_files, verify_public_files


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def separate_output(value):
    output = Path(value).expanduser().resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError("Use an output directory outside the public distribution")
    return output


def run_python(script, *arguments):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True, env=os.environ.copy())


def copy_workspace(output):
    if output.exists():
        raise ValueError("Replay output directory already exists; select a new directory")
    shutil.copytree(ROOT, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def compare_numerical_outputs(reference, replay, include_scores=False):
    import numpy as np
    paths = sorted(p.relative_to(reference).as_posix() for p in (reference / "results").glob("*.csv"))
    paths += sorted(p.relative_to(reference).as_posix() for p in (reference / "results").glob("*.npz"))
    if include_scores:
        paths += sorted(p.relative_to(reference).as_posix() for p in (reference / "scores").glob("*.npz"))
    arrays = 0
    byte_identical = 0
    for relative in paths:
        source, target = reference / relative, replay / relative
        if sha(source) == sha(target):
            byte_identical += 1
        if source.suffix == ".npz":
            with np.load(source, allow_pickle=False) as a, np.load(target, allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError("Replayed array keys differ: " + relative)
                for key in a.files:
                    if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                        raise ValueError("Replayed array differs: " + relative + ":" + key)
                    arrays += 1
        elif source.read_bytes() != target.read_bytes():
            raise ValueError("Replayed result table differs: " + relative)
    return {"status": "PASS", "files_compared": len(paths), "byte_identical_files": byte_identical,
            "npz_arrays_exactly_equal": arrays, "includes_799_reconstructed_score_files": include_scores,
            "reference": "public distribution archived numerical outputs", "replay": "separate caller-selected directory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-files")
    for name in ("audit", "synthetic", "recalculate", "rebuild"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in ("audit", "rebuild"):
            command.add_argument("--source-root", type=Path, required=name == "rebuild")
        if name == "rebuild":
            command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    release_check = verify_public_files(ROOT)
    if args.command == "verify-files":
        print(json.dumps(release_check, indent=2))
        return
    output = separate_output(args.output)
    if args.command == "audit":
        command = ["--root", ROOT, "--output", output]
        if args.source_root is not None:
            command += ["--source-root", args.source_root.resolve()]
        run_python(ROOT / "audit/verify_coupling_results.py", *command)
        return
    if args.command == "synthetic":
        run_python(ROOT / "audit/run_synthetic_portable.py", "--root", ROOT, "--output", output)
        return
    copy_workspace(output)
    if args.command == "rebuild":
        shutil.rmtree(output / "scores")
        shutil.rmtree(output / "results")
        (output / "scores").mkdir()
        (output / "results").mkdir()
        # The original numeric implementation is loaded unchanged. The one
        # substituted callback performs the documented public hash mapping.
        sys.path.insert(0, str(output / "code"))
        runner = import_file("run_coupling", output / "code/run_coupling.py")
        runner.verify_code_lock = verify_frozen_files
        for stage in ("calibration", "evaluation"):
            old_argv = sys.argv
            try:
                sys.argv = [str(output / "code/run_coupling.py"), "--root", str(output),
                            "--input-root", str(args.source_root.resolve()), "--stage", stage,
                            "--workers", str(args.workers)]
                runner.main()
            finally:
                sys.argv = old_argv
            run_python(output / "code/analyze_coupling.py", "calibrate" if stage == "calibration" else "evaluate", "--root", output)
    else:
        run_python(output / "code/analyze_coupling.py", "calibrate", "--root", output)
        run_python(output / "code/analyze_coupling.py", "evaluate", "--root", output)
    run_python(output / "code/noise_diagnostics.py", "--root", output)
    comparison = compare_numerical_outputs(ROOT, output, include_scores=args.command == "rebuild")
    (output / "PUBLIC_REPLAY_COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    run_python(output / "audit/verify_coupling_results.py", "--root", output, "--output", output / "replay_audit",
               *(["--source-root", args.source_root.resolve()] if args.command == "rebuild" else []))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
