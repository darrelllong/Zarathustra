"""SSH dispatcher for LANL cache-surface chunk selector runs on /tiamat hosts.

This is the chunk-surface analogue of `altgan.ssh_tracebootstrap_shuffle_pack`:
it uses SSH (RSA key) to run `altgan.launch_chunk_surface_multiseed` on a host
that has `/tiamat/zarathustra` mounted, then optionally commits + pushes the
updated LANL docs through git (no scp).

Typical use (from any machine with SSH access):

  python3 -m altgan.ssh_chunk_surface_multiseed \\
    --host baase \\
    --tmux-session tw_r308_refine \\
    -- \\
    --real /tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv \\
    --base-template "/tiamat/zarathustra/altgan-output/twitter_chunksurf_r307_refine16_ck16384_seed{seed}_fake_1000k.csv" \\
    --donor-globs "/tiamat/zarathustra/altgan-output/twitter_chunksurf_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/twitter*_seed42_fake_*.csv" \\
    --tag-prefix twitter_chunksurf_r308_refine8 \\
    --pipeline 8192,4096 \\
    --max-accepts 8 \\
    --max-evals 250 \\
    --emit-markdown \\
    --append-markdown RESPONSE-LANL.md,altgan/RESULTS.md

Use `--sync bundle` (default) to stream the *local* `main` branch as a git
bundle over SSH and hard-reset the remote repo before running.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="baase", help="SSH target (e.g. baase, baase.local, vinge.local).")
    p.add_argument("--user", default=None, help="Optional SSH user (default: ssh config).")
    p.add_argument("--ssh-key", default="~/.ssh/id_rsa", help="Path to RSA key (default: ~/.ssh/id_rsa).")
    p.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra ssh -o option (repeatable).",
    )
    p.add_argument("--no-agent-forwarding", action="store_true", help="Disable ssh agent forwarding (-A).")
    p.add_argument(
        "--no-proxyjump",
        action="store_true",
        help="Disable ssh ProxyJump by passing `-o ProxyJump=none`.",
    )
    p.add_argument("--repo-dir", default="~/LANL/Zarathustra", help="Repo path on remote host.")
    p.add_argument(
        "--sync",
        choices=["pull", "bundle", "none"],
        default="bundle",
        help="How to sync the remote repo before running.",
    )
    p.add_argument("--ref", default="main", help="Local git ref to sync when using --sync bundle.")
    p.add_argument(
        "--tmux-session",
        default=None,
        help="If set, run the command in a detached tmux session with this name.",
    )
    p.add_argument("--commit", action="store_true", help="Commit updated LANL docs on the remote host (after the run).")
    p.add_argument("--push", action="store_true", help="Push remote commit to origin/main (requires --commit).")
    p.add_argument("--commit-message", default="LANL: chunk-surface update", help="Remote commit message (used only with --commit).")
    p.add_argument("--dry-run", action="store_true", help="Print the ssh command and exit.")
    p.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments passed to `python3 -m altgan.launch_chunk_surface_multiseed` after `--`.",
    )
    ns = p.parse_args(argv)
    remainder = list(ns.remainder)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    return ns, remainder


def _q(value: str) -> str:
    return shlex.quote(value)


def _remote_repo_dir_expr(repo_dir: str) -> str:
    repo_dir = repo_dir.strip()
    if repo_dir == "~":
        return '"$HOME"'
    if repo_dir.startswith("~/"):
        return f"\"$HOME/{repo_dir[2:]}\""
    return _q(repo_dir)


def _ssh_argv(*, args: argparse.Namespace) -> list[str]:
    host = args.host if args.user is None else f"{args.user}@{args.host}"
    key = str(Path(args.ssh_key).expanduser())
    ssh_argv = [
        "ssh",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
    ]
    if args.no_proxyjump:
        ssh_argv.extend(["-o", "ProxyJump=none"])
    for opt in args.ssh_option:
        ssh_argv.extend(["-o", opt])
    if not args.no_agent_forwarding:
        ssh_argv.append("-A")
    ssh_argv.extend([host, "--"])
    return ssh_argv


def _create_local_bundle_bytes(*, ref: str) -> bytes:
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    proc = subprocess.run(
        ["git", "bundle", "create", "-q", "-", ref],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(f"git bundle create failed with code {proc.returncode}")
    return proc.stdout


def _build_remote_bundle_sync_script(*, args: argparse.Namespace, ref: str) -> str:
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    remote_ref = f"refs/remotes/codex_sync/{ref}"
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo_dir_expr}",
            "bundle_path=$(mktemp)",
            "cat > \"$bundle_path\"",
            f"git fetch \"$bundle_path\" {shlex.quote(ref)}:{shlex.quote(remote_ref)}",
            "rm -f \"$bundle_path\"",
            f"git checkout -B {shlex.quote(ref)} {shlex.quote(remote_ref)}",
            "git clean -fdx",
            "echo '[ssh_chunk_surface] Remote repo synced via git bundle.'",
        ]
    )


def _build_remote_pull_sync_script(*, args: argparse.Namespace, ref: str) -> str:
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo_dir_expr}",
            f"git checkout -B {shlex.quote(ref)} origin/{shlex.quote(ref)} || git checkout {shlex.quote(ref)}",
            "git pull --rebase origin main",
            "git clean -fdx",
            "echo '[ssh_chunk_surface] Remote repo synced via git pull.'",
        ]
    )


def _build_remote_run_script(
    *,
    args: argparse.Namespace,
    launch_args: list[str],
) -> str:
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    cmd = ["python3", "-u", "-m", "altgan.launch_chunk_surface_multiseed"] + launch_args
    cmd_str = " ".join(_q(part) for part in cmd)
    lines = ["set -euo pipefail", f"cd {repo_dir_expr}"]
    if args.tmux_session:
        session = _q(args.tmux_session)
        # We always run tmux jobs detached so they are explicitly managed.
        lines += [
            f"tmux new-session -d -s {session} {shlex.quote(cmd_str)}",
            f"echo '[ssh_chunk_surface] tmux session started: {args.tmux_session}'",
        ]
        return "\n".join(lines)
    lines.append(cmd_str)
    if args.commit:
        lines += [
            "echo '[ssh_chunk_surface] committing updated docs...'",
            "git status --porcelain",
            "git add -A",
            f"git commit -m {_q(args.commit_message)}",
        ]
        if args.push:
            lines.append("git push origin HEAD:main")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args, launch_args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not launch_args:
        raise SystemExit("missing launch args: pass `-- --real ... --base-template ...`")
    if args.push and not args.commit:
        raise SystemExit("--push requires --commit")

    ssh_argv = _ssh_argv(args=args)

    scripts: list[tuple[str, bytes | None]] = []
    if args.sync == "bundle":
        bundle = _create_local_bundle_bytes(ref=args.ref)
        scripts.append((_build_remote_bundle_sync_script(args=args, ref=args.ref), bundle))
    elif args.sync == "pull":
        scripts.append((_build_remote_pull_sync_script(args=args, ref=args.ref), None))

    scripts.append((_build_remote_run_script(args=args, launch_args=launch_args), None))

    # Execute each stage as a separate SSH invocation so bundle bytes can be piped cleanly.
    for script, stdin_bytes in scripts:
        full = ssh_argv + ["bash", "-lc", script]
        print("+ " + " ".join(_q(part) for part in full), flush=True)
        if args.dry_run:
            if stdin_bytes is not None:
                print(f"[ssh_chunk_surface] dry-run: would stream {len(stdin_bytes)} bundle bytes", flush=True)
            continue
        subprocess.run(full, input=stdin_bytes, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

