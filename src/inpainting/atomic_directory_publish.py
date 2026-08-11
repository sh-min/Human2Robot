"""Transactional publication of a completed directory tree."""
from __future__ import annotations

import os
import shutil
import signal
import tempfile


def _raise_on_sigterm(signum, _frame):
    """Turn SIGTERM into an exception so the publication can roll back."""
    raise SystemExit(128 + signum)


def publish_directory(staging_dir, final_dir):
    """Replace ``final_dir`` with a complete sibling ``staging_dir``.

    The old directory is restored if an exception, Ctrl-C, or SIGTERM arrives
    after it has been moved aside but before the staging rename commits.
    """
    staging_dir = os.path.abspath(os.fspath(staging_dir))
    final_dir = os.path.abspath(os.fspath(final_dir))
    parent = os.path.dirname(final_dir)
    if os.path.dirname(staging_dir) != parent:
        raise ValueError("staging and final directories must be siblings")
    if not os.path.isdir(staging_dir) or os.path.islink(staging_dir):
        raise ValueError(f"staging directory is missing or invalid: {staging_dir}")
    if os.path.lexists(final_dir) and (
        not os.path.isdir(final_dir) or os.path.islink(final_dir)
    ):
        raise ValueError(f"final path is not a real directory: {final_dir}")

    backup_dir = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        try:
            if os.path.exists(final_dir):
                backup_dir = tempfile.mkdtemp(
                    prefix=f".{os.path.basename(final_dir)}.backup.",
                    dir=parent,
                )
                os.rmdir(backup_dir)
                os.replace(final_dir, backup_dir)
            os.replace(staging_dir, final_dir)
        except BaseException:
            if backup_dir is not None and os.path.exists(backup_dir):
                if not os.path.lexists(final_dir):
                    os.replace(backup_dir, final_dir)
                elif not os.path.lexists(staging_dir):
                    # The staging rename committed before the signal was
                    # delivered.  The new directory is already complete.
                    shutil.rmtree(backup_dir, ignore_errors=True)
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)
