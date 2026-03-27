"""
Trace loading and preprocessing for LLGAN.

All fields are normalized to [-1, 1].  Opcode (read/write) is binarized to
+1 / -1.  String/categorical fields are label-encoded then scaled.

Timestamp columns are delta-encoded (inter-arrival times) rather than stored
as absolute values.  This makes the distribution stationary and far easier for
an LSTM to model.  Inverse-transform reconstructs absolute times via cumsum.

Supported formats
-----------------
spc          : ASU, LBA, Size(512B), Opcode(r/w), Timestamp(s)
msr          : Timestamp(100ns), Hostname, DiskNumber, Type, Offset, Size, ResponseTime
k5cloud      : StorageVolumeId, Timestamp, Opcode(R/W), Offset, Size(1024B)
systor       : Timestamp, ResponseTime, Opcode(R/W), LUN, Offset, Size(512B)
oracle_general: libCacheSim binary format (24 bytes/record):
                uint32 ts, uint64 obj_id, uint32 obj_size,
                int32 next_access_vtime, int16 op, int16 tenant_id
exchange_etw : Windows ETW .csv.gz (MSR Exchange Server traces, SNIA IOTTA):
               DiskRead/DiskWrite completion events only;
               ts(µs), opcode, offset(bytes), size(bytes), response_time(µs), disk
csv          : generic — infer numeric cols; opcode col named 'opcode'/'type'/'rw'
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Format readers  →  returns a raw DataFrame with canonical column names
# ---------------------------------------------------------------------------

def _read_spc(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python",
                     names=["asu", "lba", "size", "opcode", "ts"],
                     nrows=max_records)
    return df


def _read_msr(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ts", "hostname", "disk", "opcode",
                             "offset", "size", "response_time"],
                     nrows=max_records)
    df["ts"] = df["ts"] / 1e7          # 100ns → seconds
    df = df.drop(columns=["hostname"])
    return df


def _read_k5cloud(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["volume_id", "ts", "opcode", "offset", "size"],
                     nrows=max_records)
    return df


def _read_systor(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ts", "response_time", "opcode", "lun",
                             "offset", "size"],
                     nrows=max_records)
    return df


def _read_oracle_general(path: str, max_records: int) -> pd.DataFrame:
    """
    libCacheSim oracleGeneral binary format.
    24 bytes per record: uint32 ts | uint64 obj_id | uint32 obj_size |
                         int32 next_access_vtime | int16 op | int16 tenant_id
    op: 0=read, 1=write; next_access_vtime is a forward pointer (record index
    of next access to same object) — derived annotation, not a workload field.
    """
    import subprocess

    n_bytes = max_records * 24  # each record is exactly 24 bytes

    if path.endswith(".zst"):
        # Stream-decompress only the bytes we need — much faster than reading
        # the full file when max_records << total records.
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    elif path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as f:
            raw = f.read(n_bytes)
    else:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)

    dt = np.dtype([
        ("ts",       "<u4"),
        ("obj_id",   "<u8"),
        ("obj_size", "<u4"),
        ("vtime",    "<i4"),
        ("op",       "<i2"),
        ("tenant",   "<i2"),
    ])
    n = min(len(raw) // 24, max_records)
    arr = np.frombuffer(raw[:n * 24], dtype=dt)
    df = pd.DataFrame(arr)
    df.drop(columns=["vtime"], inplace=True)
    df.rename(columns={"op": "opcode"}, inplace=True)
    return df


def _read_csv(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_records)
    return df


def _read_exchange_etw(path: str, max_records: int) -> pd.DataFrame:
    """
    Microsoft Exchange Server ETW disk trace (MSR SNIA dataset).
    Format: Windows ETW .csv.gz with mixed event types; we extract only
    DiskRead/DiskWrite completion events (not DiskReadInit/DiskWriteInit).

    Fields extracted:
      ts            – TimeStamp in microseconds from trace start (→ delta-encoded IAT)
      opcode        – 'r' (DiskRead) or 'w' (DiskWrite)
      offset        – ByteOffset in bytes (hex → int)
      size          – IOSize in bytes (hex → int, log-transformed)
      response_time – ElapsedTime in microseconds (log-transformed)
      disk          – DiskNum (integer LUN id)
    """
    import gzip
    import re

    # Matches DiskRead/DiskWrite completion lines (but NOT DiskReadInit/DiskWriteInit).
    # Columns: EventType, TimeStamp, ProcessName(PID), ThreadID, IrpPtr,
    #          ByteOffset, IOSize, ElapsedTime, DiskNum, IrpFlags, ...
    pattern = re.compile(
        r'^\s*(DiskRead|DiskWrite),\s+(\d+),\s+[^,]+,\s+\d+,\s+\S+,\s+'
        r'(0x[\da-fA-F]+),\s+(0x[\da-fA-F]+),\s+(\d+),\s+(\d+),'
    )

    rows = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        for line in f:
            if len(rows) >= max_records:
                break
            m = pattern.match(line)
            if m:
                event_type, ts, offset_hex, size_hex, elapsed, disk = m.groups()
                rows.append({
                    'ts':            int(ts),
                    'opcode':        'r' if event_type == 'DiskRead' else 'w',
                    'offset':        int(offset_hex, 16),
                    'size':          int(size_hex, 16),
                    'response_time': int(elapsed),
                    'disk':          int(disk),
                })

    if not rows:
        raise ValueError(f"No DiskRead/DiskWrite events found in {path}")

    return pd.DataFrame(rows)


_READERS = {
    "spc": _read_spc,
    "msr": _read_msr,
    "k5cloud": _read_k5cloud,
    "systor": _read_systor,
    "oracle_general": _read_oracle_general,
    "exchange_etw": _read_exchange_etw,
    "csv": _read_csv,
}

# Timestamp column names — will be delta-encoded
_TS_COLS = {"ts", "timestamp"}

# Columns whose raw values are read/write opcodes
_OPCODE_COLS = {"opcode", "type", "rw", "op"}

# Columns that are log-transformed before min-max normalization.
# These have heavy-tailed / power-law distributions that confound tanh output.
# ts_delta (inter-arrival times) and obj_size are the canonical cases.
_LOG_COLS = {"ts", "timestamp", "obj_size", "size", "response_time"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TracePreprocessor:
    """
    Fit on training data; transform train + val to [-1, 1].

    Timestamp columns are delta-encoded before normalization so the model
    learns inter-arrival time distributions rather than absolute time values.
    The first absolute timestamp is stored for reconstruction.
    """

    def __init__(self):
        self.col_names: List[str] = []
        self.num_cols: int = 0
        self._stats: Dict[str, dict] = {}
        self._cat_maps: Dict[str, dict] = {}
        self._delta_cols: List[str] = []     # columns that were delta-encoded
        self._first_vals: Dict[str, float] = {}  # for inverse cumsum
        self._log_cols: List[str] = []       # columns that are log1p-transformed

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TracePreprocessor":
        df = df.copy()
        self._delta_cols = [c for c in df.columns if c.lower() in _TS_COLS]
        df = self._apply_deltas(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        # Identify log-transform columns: heavy-tailed continuous columns.
        # Applied after delta-encoding so ts_delta (inter-arrival times) is
        # also log-transformed, which is the right thing — IAT is power-law.
        self._log_cols = [
            c for c in df.columns
            if c.lower() in _LOG_COLS and c not in self._cat_maps
        ]
        df = self._apply_log(df)
        self.col_names = list(df.columns)
        self.num_cols = len(self.col_names)
        for col in self.col_names:
            lo = float(df[col].min())
            hi = float(df[col].max())
            self._stats[col] = {"lo": lo, "hi": hi}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df = self._apply_deltas(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._apply_log(df)
        out = np.zeros((len(df), self.num_cols), dtype=np.float32)
        for i, col in enumerate(self.col_names):
            lo = self._stats[col]["lo"]
            hi = self._stats[col]["hi"]
            span = hi - lo if hi != lo else 1.0
            out[:, i] = (df[col].values.astype(np.float32) - lo) / span * 2 - 1
        return out

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        data = {}
        for i, col in enumerate(self.col_names):
            lo = self._stats[col]["lo"]
            hi = self._stats[col]["hi"]
            span = hi - lo if hi != lo else 1.0
            vals = (arr[:, i].astype(np.float64) + 1) / 2 * span + lo
            # Undo log1p transform before any other post-processing
            if col in self._log_cols:
                vals = np.expm1(np.clip(vals, 0, None))
            if col in self._cat_maps:
                inv_map = {v: k for k, v in self._cat_maps[col].items()}
                vals = np.array([inv_map.get(int(round(v)), v) for v in vals])
            data[col] = vals

        df = pd.DataFrame(data)

        # Undo delta encoding: cumsum of deltas + start restores absolute timestamps.
        # The first delta is 0 by construction (_apply_deltas uses prepend=vals[0]),
        # so cumsum gives [0, d1, d1+d2, ...] and adding start gives the correct series.
        # Previous code did concatenate([[start], deltas[:-1]]).cumsum() which shifted
        # the series by one position and duplicated the first timestamp — now fixed.
        for col in self._delta_cols:
            if col in df.columns:
                start = self._first_vals.get(col, 0.0)
                df[col] = start + np.cumsum(df[col].values)

        return df

    # ------------------------------------------------------------------
    def _apply_deltas(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self._delta_cols:
            if col not in df.columns:
                continue
            vals = df[col].values.astype(np.float64)
            if fit:
                self._first_vals[col] = float(vals[0])
            deltas = np.diff(vals, prepend=vals[0])   # first delta = 0
            # Clip negative deltas (out-of-order timestamps) to 0
            deltas = np.clip(deltas, 0, None)
            df = df.copy()
            df[col] = deltas
        return df

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """log1p-transform heavy-tailed columns to approximately Gaussianize them."""
        df = df.copy()
        for col in self._log_cols:
            if col in df.columns:
                df[col] = np.log1p(np.clip(df[col].values.astype(np.float64), 0, None))
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col.lower() in _OPCODE_COLS:
                df[col] = self._encode_opcode(df[col])
            elif df[col].dtype == object:
                df[col] = self._encode_labels(col, df[col], fit)
        return df.apply(pd.to_numeric, errors="coerce").fillna(0)

    @staticmethod
    def _encode_opcode(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        def _map(x):
            if x in {"r", "read", "0"}:
                return 1.0
            elif x in {"w", "write", "1"}:
                return -1.0
            else:
                try:
                    return 1.0 if int(float(x)) == 0 else -1.0
                except ValueError:
                    return 0.0
        return s.map(_map)

    def _encode_labels(self, col: str, series: pd.Series, fit: bool) -> pd.Series:
        if fit:
            unique = sorted(series.dropna().unique())
            self._cat_maps[col] = {v: i for i, v in enumerate(unique)}
        m = self._cat_maps.get(col, {})
        return series.map(lambda x: m.get(x, 0))


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TraceDataset(Dataset):
    """
    Sliding-window dataset over a normalized trace array.
    Each item is a (timestep, num_cols) float32 tensor in [-1, 1].
    """

    def __init__(self, data: np.ndarray, timestep: int):
        self.data = torch.from_numpy(data)
        self.timestep = timestep
        self.n_windows = max(0, len(data) - timestep)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx: idx + self.timestep]


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------

def load_trace(
    path: str,
    fmt: str,
    max_records: int,
    timestep: int,
    train_split: float = 0.8,
) -> Tuple[TraceDataset, TraceDataset, TracePreprocessor]:
    """Load, preprocess, and split a trace file."""
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown trace format '{fmt}'. "
                         f"Choose from: {list(_READERS)}")

    # oracle_general handles its own decompression; others go through pandas
    if fmt != "oracle_general" and str(path).endswith(".zst"):
        import subprocess, tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        subprocess.run(["zstd", "-d", path, "-o", tmp.name, "-f"], check=True)
        df = reader(tmp.name, max_records)
        os.unlink(tmp.name)
    else:
        df = reader(str(path), max_records)

    n_train = int(len(df) * train_split)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:].reset_index(drop=True)

    prep = TracePreprocessor()
    prep.fit(df_train)

    train_arr = prep.transform(df_train)
    val_arr   = prep.transform(df_val)

    train_ds = TraceDataset(train_arr, timestep)
    val_ds   = TraceDataset(val_arr,   timestep)

    return train_ds, val_ds, prep
