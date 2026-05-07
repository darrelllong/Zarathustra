"""SSH dispatcher for TraceBootstrap shuffle pack runs on /tiamat hosts.

This script exists to make the LANL TraceBootstrap (shuffle) ledger refresh
repeatable from a laptop/CI environment: it uses SSH (RSA key) to run the
standard `altgan.launch_trace_bootstrap_shuffle_pack` command on a host that has
`/tiamat/zarathustra` mounted, then optionally commits + pushes the updated
LANL docs through git (no scp).

Typical use (from any machine with SSH access):

  python3 -m altgan.ssh_tracebootstrap_shuffle_pack --host baase.local

If `/tiamat` is mounted elsewhere on the remote host, pass `--zarathustra-root`.
Use `--tmux-session` for long runs so the job is explicitly managed.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--host",
        default="baase.local",
        help="SSH target (e.g. baase, baase.local, or vinge.local).",
    )
    p.add_argument("--user", default=None, help="Optional SSH user (default: ssh config).")
    p.add_argument("--ssh-key", default="~/.ssh/id_rsa", help="Path to RSA key (default: ~/.ssh/id_rsa).")
    p.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra ssh -o option (repeatable), e.g. --ssh-option StrictHostKeyChecking=accept-new",
    )
    p.add_argument(
        "--no-agent-forwarding",
        action="store_true",
        help="Disable ssh agent forwarding (-A).",
    )
    p.add_argument(
        "--no-proxyjump",
        action="store_true",
        help=(
            "Disable ssh ProxyJump (overrides any ssh config) by passing "
            "`-o ProxyJump=none`. Useful when your ssh config routes `baase` via "
            "`vinge` and DNS for the jump host is unavailable."
        ),
    )
    p.add_argument("--repo-dir", default="~/LANL/Zarathustra", help="Repo path on remote host.")
    p.add_argument(
        "--zarathustra-root",
        default="/tiamat/zarathustra",
        help="Remote root containing traces/ and llgan-output/ (default: /tiamat/zarathustra).",
    )
    p.add_argument(
        "--sync",
        choices=["pull", "bundle", "none"],
        default="bundle",
        help=(
            "How to sync the remote repo before running: "
            "`pull` runs `git pull --rebase origin main` on the remote; "
            "`bundle` streams the *local* `main` branch as a git bundle over SSH and hard-resets the remote; "
            "`none` skips syncing."
        ),
    )
    p.add_argument(
        "--corpora",
        nargs="+",
        default=["twitter", "metakv", "metacdn", "wiki"],
        help=(
            "Comma-separated or space-separated corpus list. Examples: "
            "`--corpora twitter metakv` or `--corpora twitter,metakv`."
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        default=["42", "80", "81", "82"],
        help="Comma-separated or space-separated seed list (default: 42,80,81,82).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Remote output root (default: <zarathustra-root>/altgan-output).",
    )
    p.add_argument(
        "--emit-dir",
        default=None,
        help="Remote paste-ready directory (default: <output-root>/paste_ready).",
    )
    p.add_argument(
        "--tmux-session",
        default=None,
        help="If set, run the command in a detached tmux session with this name.",
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="Commit updated LANL docs on the remote host (after the run).",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Push remote commit to origin/main (requires --commit).",
    )
    p.add_argument(
        "--commit-message",
        default="LANL: publish TraceBootstrap shuffle panels",
        help="Remote commit message (used only with --commit).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the ssh command and exit.")
    return p.parse_args()


def _normalize_csv_list(values: list[str]) -> str:
    parts: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parts.append(part)
    return ",".join(parts)


def _q(value: str) -> str:
    return shlex.quote(value)


def _remote_repo_dir_expr(repo_dir: str) -> str:
    """Return a shell-safe `cd ...` expression for a remote repo dir.

    We intentionally avoid single-quoting a leading `~` because that prevents
    tilde expansion on the remote shell.
    """
    repo_dir = repo_dir.strip()
    if repo_dir == "~":
        return '"$HOME"'
    if repo_dir.startswith("~/"):
        # Use $HOME so it expands on the remote host even when quoted.
        return f"\"$HOME/{repo_dir[2:]}\""
    return _q(repo_dir)


def _remote_shell(*, args: argparse.Namespace) -> str:
    host = args.host if args.user is None else f"{args.user}@{args.host}"
    key = str(Path(args.ssh_key).expanduser())
    ssh_parts = [
        "ssh",
        "-i",
        _q(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
    ]
    if args.no_proxyjump:
        ssh_parts.extend(["-o", "ProxyJump=none"])
    for opt in args.ssh_option:
        ssh_parts.extend(["-o", _q(opt)])
    if not args.no_agent_forwarding:
        ssh_parts.append("-A")
    ssh_parts.extend([_q(host), "--"])
    return " ".join(
        ssh_parts
    )


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
    # `git bundle create - <ref>` writes the bundle to stdout.
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


def _build_remote_sync_script(*, args: argparse.Namespace, ref: str) -> str:
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    remote_ref = f"refs/remotes/codex_sync/{ref}"
    script_lines = [
        "set -euo pipefail",
        f"cd {repo_dir_expr}",
        "bundle_path=$(mktemp)",
        "cat > \"$bundle_path\"",
        f"git fetch \"$bundle_path\" {shlex.quote(ref)}:{shlex.quote(remote_ref)}",
        "rm -f \"$bundle_path\"",
        f"git checkout -B {shlex.quote(ref)} {shlex.quote(remote_ref)}",
        "git clean -fdx",
        "echo '[ssh_tracebootstrap] Remote repo synced via git bundle.'",
    ]
    return "\n".join(script_lines)


def _build_remote_script(*, args: argparse.Namespace) -> str:
    zar_root = args.zarathustra_root
    output_root = args.output_root or f"{zar_root}/altgan-output"
    emit_dir = args.emit_dir or f"{output_root}/paste_ready"
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    corpora = _normalize_csv_list(args.corpora)
    seeds = _normalize_csv_list(args.seeds)
    cmd = [
        "python3",
        "-m",
        "altgan.launch_trace_bootstrap_shuffle_pack",
        "--corpora",
        corpora,
        "--seeds",
        seeds,
        "--zarathustra-root",
        zar_root,
        "--output-root",
        output_root,
        "--update-lanl-docs",
        "--markdown",
        "--skip-existing",
        "--keep-going",
        "--emit-markdown-dir",
        emit_dir,
        "--emit-summary-json-dir",
        emit_dir,
    ]
    if args.commit and args.push:
        push_line = "git push origin main"
    elif args.commit:
        push_line = "echo '[ssh_tracebootstrap] NOTE: --push not set; leaving commit unpushed.'"
    else:
        push_line = "echo '[ssh_tracebootstrap] NOTE: --commit not set; leaving docs uncommitted.'"

    commit_block = ""
    if args.commit:
        commit_msg = args.commit_message
        commit_block = "\n".join(
            [
                "git status --porcelain",
                "git add altgan/RESULTS.md RESPONSE-LANL.md",
                # Only commit if there is something staged/changed.
                "if git diff --cached --quiet && git diff --quiet; then",
                "  echo '[ssh_tracebootstrap] No doc changes detected; skipping commit.'",
                "else",
                f"  git commit -m {_q(commit_msg)} || true",
                "fi",
            ]
        )

    script_lines = [
        "set -euo pipefail",
        f"cd {repo_dir_expr}",
        *(["git pull --rebase origin main"] if args.sync == "pull" else []),
        "python3 -V",
        "echo '[ssh_tracebootstrap] Running TraceBootstrap shuffle pack...'",
        " ".join(_q(part) for part in cmd),
    ]
    if commit_block:
        script_lines.append("echo '[ssh_tracebootstrap] Committing doc updates...'")
        script_lines.append(commit_block)
    script_lines.append(push_line)
    script_lines.append("echo '[ssh_tracebootstrap] Done.'")
    return "\n".join(script_lines)


def main() -> int:
    args = _parse_args()
    if args.push and not args.commit:
        raise SystemExit("--push requires --commit.")

    remote_script = _build_remote_script(args=args)
    ssh_prefix = _remote_shell(args=args)
    remote_cmd = f"{ssh_prefix} bash -lc {_q(remote_script)}"

    if args.tmux_session:
        session = args.tmux_session
        # Run the remote command in tmux so the job is explicit + managed.
        tmux_cmd = (
            f"{ssh_prefix} bash -lc "
            f"{_q(f'tmux new-session -d -s {shlex.quote(session)} bash -lc {shlex.quote(remote_script)}')}"
        )
        remote_cmd = tmux_cmd

    if args.dry_run:
        if args.sync == "bundle":
            sync_script = _build_remote_sync_script(args=args, ref="main")
            sync_cmd = (
                f"git bundle create -q - main | "
                f"{ssh_prefix} bash -lc {_q(sync_script)}"
            )
            print("# bundle sync:")
            print(sync_cmd)
            print("# run:")
        print(remote_cmd)
        if args.tmux_session:
            key = str(Path(args.ssh_key).expanduser())
            host = args.host if args.user is None else f"{args.user}@{args.host}"
            attach_parts: list[str] = ["ssh", "-i", key]
            for opt in args.ssh_option:
                attach_parts.extend(["-o", opt])
            if not args.no_agent_forwarding:
                attach_parts.append("-A")
            attach_parts.extend([host, "tmux", "attach", "-t", args.tmux_session])
            print("# attach: " + " ".join(shlex.quote(part) for part in attach_parts))
        return 0

    print(f"[ssh_tracebootstrap] dispatch -> {args.host}", flush=True)
    if args.sync == "bundle":
        bundle_bytes = _create_local_bundle_bytes(ref="main")
        sync_script = _build_remote_sync_script(args=args, ref="main")
        ssh_argv = _ssh_argv(args=args)
        # 1) Sync the remote repo to the local `main` tip via bundle streamed over SSH.
        sync_res = subprocess.run(
            [*ssh_argv, "bash", "-lc", sync_script],
            input=bundle_bytes,
        )
        if sync_res.returncode != 0:
            return sync_res.returncode
        # 2) Run the actual pack command on the now-synced remote repo (tmux or direct).
        run_res = subprocess.run(remote_cmd, shell=True)
        return run_res.returncode

    return subprocess.run(remote_cmd, shell=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
