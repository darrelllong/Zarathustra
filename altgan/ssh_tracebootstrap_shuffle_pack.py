"""SSH dispatcher for TraceBootstrap shuffle pack runs on /tiamat hosts.

This script exists to make the LANL TraceBootstrap (shuffle) ledger refresh
repeatable from a laptop/CI environment: it uses SSH (RSA key) to run the
standard `altgan.launch_trace_bootstrap_shuffle_pack` command on a host that has
`/tiamat/zarathustra` mounted, then optionally commits + pushes the updated
LANL docs through git (no scp).

Typical use (from any machine with SSH access):

  python3 -m altgan.ssh_tracebootstrap_shuffle_pack --host vinge.local

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="vinge.local", help="SSH target (e.g. vinge.local or baase.local).")
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
    p.add_argument("--repo-dir", default="~/LANL/Zarathustra", help="Repo path on remote host.")
    p.add_argument(
        "--zarathustra-root",
        default="/tiamat/zarathustra",
        help="Remote root containing traces/ and llgan-output/ (default: /tiamat/zarathustra).",
    )
    p.add_argument("--corpora", default="twitter,metakv,metacdn,wiki")
    p.add_argument("--seeds", default="42,80,81,82")
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
    for opt in args.ssh_option:
        ssh_parts.extend(["-o", _q(opt)])
    if not args.no_agent_forwarding:
        ssh_parts.append("-A")
    ssh_parts.extend([_q(host), "--"])
    return " ".join(
        ssh_parts
    )


def _build_remote_script(*, args: argparse.Namespace) -> str:
    zar_root = args.zarathustra_root
    output_root = args.output_root or f"{zar_root}/altgan-output"
    emit_dir = args.emit_dir or f"{output_root}/paste_ready"
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    cmd = [
        "python3",
        "-m",
        "altgan.launch_trace_bootstrap_shuffle_pack",
        "--corpora",
        args.corpora,
        "--seeds",
        args.seeds,
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
        "git pull --rebase origin main",
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
    return subprocess.run(remote_cmd, shell=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
