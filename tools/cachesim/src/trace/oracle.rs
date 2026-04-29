//! oracleGeneral binary reader (zstd-compressed stream).
//!
//! Record layout matches libCacheSim's `oracleGeneral` schema:
//!     u32  clock_time
//!     u64  obj_id
//!     u32  obj_size
//!     i64  next_access_vtime   // ignored by the simulator
//! Total: 24 bytes per record, little-endian.
//!
//! Decompresses the file via `zstd::Decoder` and emits one `Access` per record.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::trace::{Access, Op};

const RECORD_SIZE: usize = 24;

pub struct OracleReader {
    inner: zstd::Decoder<'static, BufReader<File>>,
    buf: [u8; RECORD_SIZE],
    done: bool,
}

impl OracleReader {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let f = File::open(path)?;
        let dec = zstd::Decoder::new(f)?;
        Ok(Self {
            inner: dec,
            buf: [0u8; RECORD_SIZE],
            done: false,
        })
    }
}

impl Iterator for OracleReader {
    type Item = anyhow::Result<Access>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.inner.read_exact(&mut self.buf) {
            Ok(()) => {
                let ts = u32::from_le_bytes(self.buf[0..4].try_into().unwrap()) as u64;
                let obj_id = u64::from_le_bytes(self.buf[4..12].try_into().unwrap());
                let size = u32::from_le_bytes(self.buf[12..16].try_into().unwrap());
                // bytes [16..24] = next_access_vtime; intentionally unused.
                Some(Ok(Access {
                    obj_id,
                    size,
                    op: Op::Other,
                    ts,
                }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e.into()))
            }
        }
    }
}
