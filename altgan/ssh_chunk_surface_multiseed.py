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
bundle over SSH and update the remote LANL checkout before running.
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
    p.add_argument(
        "--repo-dir",
        default="~/LANL/Zarathustra",
        help="Repo path on remote host (will probe common alternates if missing).",
    )
    p.add_argument(
        "--remote-python",
        default="/tiamat/zarathustra/altgan-venv/bin/python",
        help="Python interpreter on the remote host.",
    )
    p.add_argument(
        "--remote-module",
        default="altgan.launch_chunk_surface_multiseed",
        help=(
            "Python module to execute on the remote host (default: altgan.launch_chunk_surface_multiseed). "
            "Useful for running preset wrappers like altgan.launch_baleen24_chunk_surface_multiseed."
        ),
    )
    p.add_argument(
        "--remote-log-dir",
        default="/tiamat/zarathustra/altgan-output/logs",
        help=(
            "Remote directory for stdout/stderr logs (default: /tiamat/zarathustra/altgan-output/logs). "
            "When set, the remote run redirects output to a timestamped log and prints LOG:<path>."
        ),
    )
    p.add_argument(
        "--remote-git-ssh-key",
        default="",
        help=(
            "Remote key path for git fetch/push on the remote host (optional). "
            "Leave empty to rely on whatever SSH identity/agent is already available on the remote host."
        ),
    )
    p.add_argument(
        "--remote-git-ssh-option",
        action="append",
        default=[],
        help="Extra ssh -o option for remote git operations (repeatable).",
    )
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
        help="Arguments passed to the remote module after `--`.",
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


def _remote_cd_repo_snippet(*, args: argparse.Namespace) -> list[str]:
    """Return bash lines that `cd` into the remote repo, probing common paths."""
    repo_dir_expr = _remote_repo_dir_expr(args.repo_dir)
    return [
        f"repo_dir_user={repo_dir_expr}",
        "repo_dir=''",
        'for cand in "$repo_dir_user" "$HOME/LANL/Zarathustra" "/home/darrell/LANL/Zarathustra" "$HOME/Zarathustra" "/home/darrell/Zarathustra"; do',
        '  if [ -d "$cand/.git" ]; then repo_dir="$cand"; break; fi',
        "done",
        'if [ -z "$repo_dir" ]; then',
        "  echo '[ssh_chunk_surface] ERROR: could not locate remote repo; tried:' >&2",
        '  echo "  $repo_dir_user" >&2',
        '  echo "  $HOME/LANL/Zarathustra" >&2',
        '  echo "  /home/darrell/LANL/Zarathustra" >&2',
        '  echo "  $HOME/Zarathustra" >&2',
        '  echo "  /home/darrell/Zarathustra" >&2',
        "  exit 2",
        "fi",
        'cd "$repo_dir"',
    ]


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


def _git_ssh_command_export(*, args: argparse.Namespace) -> str:
    """Return a remote-shell line exporting GIT_SSH_COMMAND for git operations.

    The remote host may rely on agent forwarding or its own on-disk key; this just
    forces BatchMode and (optionally) a specific remote key path.
    """
    cmd: list[str] = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"]
    if args.remote_git_ssh_key.strip():
        cmd.extend(["-i", args.remote_git_ssh_key.strip()])
    for opt in args.remote_git_ssh_option:
        opt = opt.strip()
        if opt:
            cmd.extend(["-o", opt])
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    # Quote the whole command string so spaces are preserved in the env var.
    return f"export GIT_SSH_COMMAND={shlex.quote(cmd_str)}"


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
    remote_ref = f"refs/remotes/codex_sync/{ref}"
    git_ssh_export = _git_ssh_command_export(args=args)
    return "\n".join(
        [
            "set -euo pipefail",
            *_remote_cd_repo_snippet(args=args),
            git_ssh_export,
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
    git_ssh_export = _git_ssh_command_export(args=args)
    return "\n".join(
        [
            "set -euo pipefail",
            *_remote_cd_repo_snippet(args=args),
            git_ssh_export,
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
    git_ssh_export = _git_ssh_command_export(args=args)
    cmd = [args.remote_python, "-u", "-m", args.remote_module] + launch_args
    cmd_str = " ".join(_q(part) for part in cmd)

    def _commit_block() -> list[str]:
        if not args.commit:
            return []
        lines2 = [
            "echo '[ssh_chunk_surface] Committing doc updates...'",
            "git status --porcelain",
            "git add altgan/RESULTS.md RESPONSE-LANL.md",
            # Only commit if there is something staged/changed.
            "if git diff --cached --quiet && git diff --quiet; then",
            "  echo '[ssh_chunk_surface] No doc changes detected; skipping commit.'",
            "else",
            f"  git commit -m {_q(args.commit_message)} || true",
            "fi",
        ]
        if args.push:
            lines2.append("git push origin HEAD:main")
        else:
            lines2.append("echo '[ssh_chunk_surface] NOTE: --push not set; leaving commit unpushed.'")
        return lines2

    def _log_wrapped_cmd() -> list[str]:
        log_dir = (args.remote_log_dir or "").strip()
        if not log_dir:
            return [cmd_str]
        safe_tag = (args.remote_module or "chunk_surface").replace("/", "_").replace(".", "_")
        return [
            f"log_dir={_q(log_dir)}",
            "mkdir -p \"$log_dir\"",
            "ts=$(date -u +%Y%m%dT%H%M%SZ)",
            f"log_path=\"$log_dir/{safe_tag}_$ts.log\"",
            f"echo \"LOG:$log_path\"",
            f"{cmd_str} > \"$log_path\" 2>&1",
        ]

    base_lines = ["set -euo pipefail", *_remote_cd_repo_snippet(args=args), git_ssh_export]
    run_lines = _log_wrapped_cmd() + _commit_block()

    if args.tmux_session:
        # For detached runs, embed the full run script (including commit/push)
        # inside the tmux session so completion triggers the git operations.
        remote_script = "\n".join(base_lines + run_lines + ["echo '[ssh_chunk_surface] Done.'"])
        session = _q(args.tmux_session)
        inner = f"tmux new-session -d -s {session} bash -lc {shlex.quote(remote_script)}"
        return "\n".join(base_lines + [inner, f"echo '[ssh_chunk_surface] tmux session started: {args.tmux_session}'"])

    return "\n".join(base_lines + run_lines + ["echo '[ssh_chunk_surface] Done.'"])


def main(argv: list[str] | None = None) -> int:
    args, launch_args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not launch_args and args.remote_module == "altgan.launch_chunk_surface_multiseed":
        raise SystemExit("missing launch args: pass `-- --real ... --base-template ...`")
    if args.push and not args.commit:
        raise SystemExit("--push requires --commit")

    def _run(ns: argparse.Namespace) -> int:
        ssh_argv = _ssh_argv(args=ns)

        scripts: list[tuple[str, bytes | None]] = []
        if ns.sync == "bundle":
            bundle = _create_local_bundle_bytes(ref=ns.ref)
            scripts.append((_build_remote_bundle_sync_script(args=ns, ref=ns.ref), bundle))
        elif ns.sync == "pull":
            scripts.append((_build_remote_pull_sync_script(args=ns, ref=ns.ref), None))

        scripts.append((_build_remote_run_script(args=ns, launch_args=launch_args), None))

        # Execute each stage as a separate SSH invocation so bundle bytes can be piped cleanly.
        for script, stdin_bytes in scripts:
            full = ssh_argv + ["bash", "-lc", script]
            print("+ " + " ".join(_q(part) for part in full), flush=True)
            if ns.dry_run:
                if stdin_bytes is not None:
                    print(
                        f"[ssh_chunk_surface] dry-run: would stream {len(stdin_bytes)} bundle bytes",
                        flush=True,
                    )
                continue
            subprocess.run(full, input=stdin_bytes, check=True)

        return 0

    try:
        return _run(args)
    except subprocess.CalledProcessError as exc:
        if args.dry_run:
            raise
        if exc.returncode != 255:
            # Non-SSH failure (remote command exited non-zero). Don't retry
            # automatically; that can be expensive and may duplicate work.
            sys.stderr.write(
                f"[ssh_chunk_surface] ERROR: remote command failed with exit code {exc.returncode}.\n"
            )
            sys.stderr.flush()
            return int(exc.returncode)
        if args.no_proxyjump:
            sys.stderr.write(
                "[ssh_chunk_surface] ERROR: ssh failed (exit 255).\n"
                "[ssh_chunk_surface] Common causes:\n"
                "  - Outbound SSH blocked by the current environment/sandbox (often shows as "
                "'Operation not permitted' or 'Connection closed by UNKNOWN port 65535').\n"
                "  - DNS not available (e.g. 'Could not resolve hostname ...').\n"
                "[ssh_chunk_surface] Fix: run this command from a machine/network that can reach the "
                "target host on port 22.\n"
            )
            sys.stderr.flush()
            return 255
        # Common failure mode: ssh-config forces ProxyJump through a host that
        # is not resolvable from the current network. Retry once without
        # ProxyJump to reduce operator friction.
        sys.stderr.write(
            "[ssh_chunk_surface] NOTE: command failed; retrying with --no-proxyjump (ProxyJump=none).\n"
        )
        sys.stderr.flush()
        retry_args = argparse.Namespace(**vars(args))
        retry_args.no_proxyjump = True
        try:
            return _run(retry_args)
        except subprocess.CalledProcessError as exc2:
            if exc2.returncode != 255:
                sys.stderr.write(
                    f"[ssh_chunk_surface] ERROR: remote command failed with exit code {exc2.returncode}.\n"
                )
                sys.stderr.flush()
                return int(exc2.returncode)
            sys.stderr.write(
                "[ssh_chunk_surface] ERROR: ssh failed (exit 255).\n"
                "[ssh_chunk_surface] Common causes:\n"
                "  - Outbound SSH blocked by the current environment/sandbox (often shows as "
                "'Operation not permitted' or 'Connection closed by UNKNOWN port 65535').\n"
                "  - DNS not available (e.g. 'Could not resolve hostname ...').\n"
                "[ssh_chunk_surface] Fix: run this command from a machine/network that can reach the "
                "target host on port 22.\n"
            )
            sys.stderr.flush()
            return 255


if __name__ == "__main__":
    raise SystemExit(main())
