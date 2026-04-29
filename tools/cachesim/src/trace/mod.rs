//! Trace abstraction. One canonical `Access` record; readers normalise to it.

pub mod csv;
pub mod oracle;

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Read,
    Write,
    Other,
}

#[derive(Debug, Clone, Copy)]
pub struct Access {
    pub obj_id: u64,
    pub size: u32,
    pub op: Op,
    pub ts: u64,
}

pub trait Trace: Iterator<Item = anyhow::Result<Access>> {}
impl<T: Iterator<Item = anyhow::Result<Access>>> Trace for T {}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    OracleGeneral,
    SyntheticCsv,
}

pub fn detect_format(path: &Path) -> anyhow::Result<Format> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name.ends_with(".zst") || name.ends_with(".oracleGeneral") {
        Ok(Format::OracleGeneral)
    } else if name.ends_with(".csv") || name.ends_with(".csv.gz") {
        Ok(Format::SyntheticCsv)
    } else {
        anyhow::bail!("cannot detect format from path: {}", path.display())
    }
}

pub fn open(path: &Path, fmt: Format) -> anyhow::Result<Box<dyn Trace>> {
    match fmt {
        Format::OracleGeneral => Ok(Box::new(oracle::OracleReader::open(path)?)),
        Format::SyntheticCsv => Ok(Box::new(csv::CsvReader::open(path)?)),
    }
}
