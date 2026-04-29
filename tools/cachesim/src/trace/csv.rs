//! Synthetic CSV reader for the outputs of `llgan/generate.py` and
//! `altgan/generate.py`.
//!
//! Both generators emit similar columns but with slight name drift. This
//! reader auto-detects the columns it needs from the header row:
//!   - obj_id     : `obj_id` | `object_id` | `block_id`
//!   - size       : `obj_size` | `size` | `bytes`
//!   - op         : `op` | `opcode` | `read_write` (optional; defaults to Read)
//!   - timestamp  : `ts` | `timestamp` | `time` (optional; defaults to row index)
//!
//! On a column-name mismatch the reader returns a clear error rather than
//! silently misaligning fields.

use std::fs::File;
use std::path::Path;

use crate::trace::{Access, Op};

pub struct CsvReader {
    inner: csv::Reader<File>,
    cols: ColIdx,
    row_ix: u64,
}

struct ColIdx {
    obj_id: usize,
    size: Option<usize>,
    op: Option<usize>,
    ts: Option<usize>,
}

impl CsvReader {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let mut rdr = csv::Reader::from_path(path)?;
        let headers = rdr.headers()?.clone();
        let cols = resolve_columns(&headers)?;
        Ok(Self {
            inner: rdr,
            cols,
            row_ix: 0,
        })
    }
}

fn resolve_columns(headers: &csv::StringRecord) -> anyhow::Result<ColIdx> {
    let find = |names: &[&str]| -> Option<usize> {
        headers
            .iter()
            .position(|h| names.iter().any(|n| h.eq_ignore_ascii_case(n)))
    };
    let obj_id = find(&["obj_id", "object_id", "block_id"])
        .ok_or_else(|| anyhow::anyhow!("CSV missing obj_id column (looked for obj_id|object_id|block_id)"))?;
    Ok(ColIdx {
        obj_id,
        size: find(&["obj_size", "size", "bytes"]),
        op: find(&["op", "opcode", "read_write"]),
        ts: find(&["ts", "timestamp", "time"]),
    })
}

fn parse_op(s: &str) -> Op {
    let lo = s.to_ascii_lowercase();
    if lo == "r" || lo == "read" || lo == "0" {
        Op::Read
    } else if lo == "w" || lo == "write" || lo == "1" {
        Op::Write
    } else {
        Op::Other
    }
}

impl Iterator for CsvReader {
    type Item = anyhow::Result<Access>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rec = csv::StringRecord::new();
        match self.inner.read_record(&mut rec) {
            Ok(true) => {
                let parse = || -> anyhow::Result<Access> {
                    let obj_id: u64 = rec
                        .get(self.cols.obj_id)
                        .ok_or_else(|| anyhow::anyhow!("missing obj_id field"))?
                        .parse()?;
                    let size: u32 = match self.cols.size {
                        Some(i) => rec.get(i).and_then(|s| s.parse().ok()).unwrap_or(0),
                        None => 0,
                    };
                    let op = match self.cols.op {
                        Some(i) => rec.get(i).map(parse_op).unwrap_or(Op::Read),
                        None => Op::Read,
                    };
                    let ts: u64 = match self.cols.ts {
                        Some(i) => rec.get(i).and_then(|s| s.parse().ok()).unwrap_or(self.row_ix),
                        None => self.row_ix,
                    };
                    Ok(Access {
                        obj_id,
                        size,
                        op,
                        ts,
                    })
                };
                let out = parse();
                self.row_ix += 1;
                Some(out)
            }
            Ok(false) => None,
            Err(e) => Some(Err(e.into())),
        }
    }
}
