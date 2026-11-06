"""Create-Dataset scan inputs and efficient, read-only result previews.

The helpers in this module are independent of Shiny so scan validation and
results conversion can be tested without starting a browser session.
"""
from __future__ import annotations

from pathlib import Path
import asyncio
import os
import threading
import sys

import pandas as pd


def parse_scan_levels(value):
    """Turn a comma-separated input into directory level labels.

    Preserve interior empty entries so scan_parameters can reject typos such
    as `site,,monitor`, rather than silently changing the directory structure.
    """
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",")]
    return [str(part).strip() for part in value]


async def choose_native_scan_folder(initial=None):
    """Request a local *server desktop* OS folder dialog when available.

    Web browser folder uploads cannot disclose a server-side absolute path.
    Running Tk in an independent process keeps its GUI lifecycle isolated from
    Shiny and avoids a blocking callback. Remote/headless servers use the
    existing in-app directory browser instead.
    """
    import os
    import asyncio
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("no desktop display is available on the Shiny server")
    initial_dir = str(Path(initial or Path.home()).expanduser().resolve())
    dialog_script = (
        "import sys, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root=tk.Tk()\nroot.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "folder=filedialog.askdirectory(parent=root, initialdir=sys.argv[1], title='Select scan folder', mustexist=True)\n"
        "print(folder, flush=True)\nroot.destroy()\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", dialog_script, initial_dir,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.communicate()
        raise RuntimeError("native folder picker timed out or was interrupted")
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(message[-1] if message else "unable to launch system folder dialog")
    selected = stdout.decode(errors="replace").strip()
    if not selected:
        return None
    folder = Path(selected).expanduser().resolve()
    if not folder.is_dir():
        raise RuntimeError("selected directory is unavailable")
    return folder


def scan_directory_choices(directory):
    """Return available immediate subfolders for the server-side folder picker.

    No files are exposed and no user-supplied commands are executed. Unreadable
    directories are skipped; callers handle permission errors on the parent.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    with os.scandir(root) as entries:
        names = []
        for entry in entries:
            try:
                if entry.is_dir():
                    names.append(entry.name)
            except OSError:
                continue
    return sorted(names, key=str.casefold)


def scan_parameters(directory, levels, utc_timezone=0, metadata=None):
    """Validate browser-supplied settings for ``Dataset.scan_folder``.

    Paths refer to the computer running Shiny (not the remote browser).
    ``scan_folder`` concatenates directory and child names, so its directory
    argument must end with the platform's path separator.
    """
    folder = str(directory or "").strip()
    if not folder:
        raise ValueError("Choose a folder before starting the scan")
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Scan folder does not exist: {path}")
    cleaned = [str(level).strip() for level in (levels or [])]
    if not cleaned or not cleaned[0]:
        raise ValueError("Enter at least one level")
    if any(not label for label in cleaned):
        raise ValueError("Levels cannot have gaps")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Level names must be unique")
    # scan_folder itself rejects certain names but currently prints an error
    # instead of raising one, so validate up front for clear UI feedback.
    reserved = {"identifier", "start", "end", "length", "datetime", "timedelta", "pm25",
                "pm25correctedrh", "pm25correctedrhandgrav", "filterid", "grav_not",
                "grav", "PM2.5", "PM10", "C02", "NO2"}
    invalid = reserved.intersection(cleaned)
    if invalid:
        raise ValueError(f"Reserved level names: {', '.join(sorted(invalid))}")
    try:
        tz = float(utc_timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError("UTC timezone must be a number") from exc
    import math
    if not math.isfinite(tz):
        raise ValueError("UTC timezone must be finite")
    add_m = dict(metadata or {})
    if any(not isinstance(k, str) or not k.strip() for k in add_m):
        raise ValueError("Metadata category names must be non-empty strings")
    return {
        "directory": str(path) + os.sep,
        "levels": cleaned,
        "utc_timezone": tz,
        "add_m": add_m,
    }


def scan_folder_dataset(directory, levels, utc_timezone=0, metadata=None, *, dataset=None, message_callback=None):
    """Scan into a supplied Dataset, creating one only for the first scan.

    The scanner appends recognized monitors to the same Dataset instance.
    Existing entries are retained on subsequent scans.
    """
    from .classes import Dataset
    params = scan_parameters(directory, levels, utc_timezone, metadata)
    if dataset is not None and not isinstance(dataset, Dataset):
        raise TypeError("Existing scan target must be a Dataset")
    result = dataset if dataset is not None else Dataset()
    if message_callback is not None:
        params["message_callback"] = message_callback
    result.scan_folder(**params)
    return result



async def scan_folder_async(directory, levels, utc_timezone=0, metadata=None, *, dataset=None, message_callback=None):
    """Scan off the Shiny event loop without its shared thread executor.

    ``asyncio.to_thread`` relies on the event loop's default executor, which
    can reject submissions while a notebook/embedded interpreter is shutting
    down. This dedicated daemon worker has no shared executor lifecycle. A
    closed event loop cannot receive results; the worker then exits quietly.
    It does *not* forcibly cancel an already-running filesystem scan.
    """
    loop = asyncio.get_running_loop()
    completed = loop.create_future()

    def deliver(success, payload):
        if completed.done():
            return
        if success:
            completed.set_result(payload)
        else:
            completed.set_exception(payload)

    def work():
        try:
            extra = {"message_callback": message_callback} if message_callback is not None else {}
            if dataset is None:
                result = scan_folder_dataset(directory, levels, utc_timezone, metadata, **extra)
            else:
                result = scan_folder_dataset(directory, levels, utc_timezone, metadata, dataset=dataset, **extra)
        except BaseException as exc:
            success, payload = False, exc
        else:
            success, payload = True, result
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(deliver, success, payload)
        except RuntimeError:
            # Session or event loop shut down during scanning.
            pass

    worker = threading.Thread(target=work, name="apmtools-folder-scan", daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        completed.cancel()
        raise RuntimeError(
            "Could not start folder scan worker; the Python process may be shutting down"
        ) from exc
    return await completed


def scan_results_frame(dataset):
    """One row per scanned monitor; never concatenate large sensor timeseries."""
    entries = []
    for identifier, monitor in dataset.items():
        raw_metadata = getattr(monitor, "m", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        row = {
            "Identifier": str(identifier),
            "Monitor": type(monitor).__name__,
            "Filename": str(metadata.get("filename", "")),
            "Rows": len(monitor) if hasattr(monitor, "__len__") else None,
        }
        for key, value in metadata.items():
            if str(key) == "filename":
                continue
            # Flatten per-monitor metadata without copying individual readings.
            label = str(key)
            if label in row:
                label = f"Metadata: {label}"
            if value is None or isinstance(value, (str, int, float, bool)):
                row[label] = value
            else:
                row[label] = str(value)
        entries.append(row)
    return pd.DataFrame(entries, columns=["Identifier", "Monitor", "Filename", "Rows"] if not entries else None)


def summary_destination(filename, *, overwrite=False):
    """Validate the on-server path for a newly generated Summary pickle."""
    name = str(filename or '').strip()
    if not name:
        raise ValueError('Enter a Summary filename')
    target = Path(name).expanduser().resolve()
    if target.suffix.lower() != '.pkl':
        raise ValueError('Summary filename must end in .pkl')
    if not target.parent.is_dir():
        raise FileNotFoundError(f'Destination directory does not exist: {target.parent}')
    if target.exists() and not overwrite:
        raise FileExistsError(f'Summary file already exists: {target}')
    if target.exists() and not target.is_file():
        raise ValueError(f'Summary destination is not a regular file: {target}')
    return target


def create_summary_from_scan(dataset, filename=None, *, all_variables=False, message_callback=None):
    """Create a Summary in memory; optionally persist a pickle for compatibility."""
    from .classes import Dataset, Summary
    from .plot_app import save_summary_pickle
    if not isinstance(dataset, Dataset):
        raise TypeError('Create summary requires a scanned Dataset')
    if not dataset:
        raise ValueError('Scan at least one monitor before creating a Summary')
    target = summary_destination(filename) if filename is not None else None
    options = {"save_csv": False, "all_variables": bool(all_variables)}
    if message_callback is not None:
        options["message_callback"] = message_callback
    result = dataset.save_summary(**options)
    if not isinstance(result, Summary):
        raise TypeError('Dataset.save_summary() did not return a Summary')
    if result.empty:
        raise ValueError('Dataset.save_summary() returned an empty Summary')
    if target is not None:
        save_summary_pickle(result, target)
    return result


async def create_summary_async(dataset, filename=None, *, all_variables=False, message_callback=None):
    """Run Summary construction off the reactive event loop in one daemon worker."""
    loop = asyncio.get_running_loop()
    completed = loop.create_future()

    def deliver(success, payload):
        if completed.done():
            return
        if success:
            completed.set_result(payload)
        else:
            completed.set_exception(payload)

    def work():
        try:
            options = {"message_callback": message_callback} if message_callback is not None else {}
            result = create_summary_from_scan(dataset, filename, all_variables=all_variables, **options)
        except BaseException as exc:
            success, payload = False, exc
        else:
            success, payload = True, result
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(deliver, success, payload)
        except RuntimeError:
            pass

    worker = threading.Thread(target=work, name='apmtools-summary-create', daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        completed.cancel()
        raise RuntimeError('Could not start Summary creation worker') from exc
    return await completed
