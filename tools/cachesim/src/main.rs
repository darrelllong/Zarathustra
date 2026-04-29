use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};

use cachesim::policy::{self, PolicyKind};
use cachesim::trace::{self, Format};

#[derive(Debug, Clone, ValueEnum)]
enum CliFormat {
    Auto,
    Oracle,
    Csv,
}

#[derive(Debug, Clone, ValueEnum)]
enum CliPolicy {
    Lru,
    Arc,
}

impl From<CliPolicy> for PolicyKind {
    fn from(p: CliPolicy) -> PolicyKind {
        match p {
            CliPolicy::Lru => PolicyKind::Lru,
            CliPolicy::Arc => PolicyKind::Arc,
        }
    }
}

/// Cache simulator (LRU, ARC) for real and synthetic I/O traces from
/// `llgan/generate.py` and `altgan/generate.py`.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Trace to simulate (the "fake" side if --real is also supplied).
    #[arg(long)]
    trace: PathBuf,

    /// Optional real trace; enables HRC-MAE(fake, real) in the report.
    #[arg(long)]
    real: Option<PathBuf>,

    /// Cache-replacement policies to simulate.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [CliPolicy::Lru, CliPolicy::Arc])]
    policy: Vec<CliPolicy>,

    /// Single cache size (mutually exclusive with --cache-sizes / --grid).
    #[arg(long)]
    cache_size: Option<usize>,

    /// Comma-separated cache sizes for an HRC sweep.
    #[arg(long, value_delimiter = ',')]
    cache_sizes: Vec<usize>,

    /// Named cache-size grid (e.g. lanl-tencent, lanl-alibaba). Loaded from
    /// `tools/cachesim/grids/<name>.json` once those files are populated.
    #[arg(long)]
    grid: Option<String>,

    /// Trace format. `auto` infers from extension.
    #[arg(long, value_enum, default_value_t = CliFormat::Auto)]
    format: CliFormat,

    /// Number of independent streams the trace represents (only used by
    /// stream-aware metrics; pass-through for now).
    #[arg(long, default_value_t = 1)]
    n_streams: u32,

    /// Random seed (reserved for future tie-breaks; deterministic by default).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Worker threads for per-cache-size parallelism (0 = rayon default).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Output JSON path (`-` writes to stdout).
    #[arg(long, default_value = "-")]
    out: String,
}

fn resolve_format(fmt: &CliFormat, path: &std::path::Path) -> anyhow::Result<Format> {
    match fmt {
        CliFormat::Auto => trace::detect_format(path),
        CliFormat::Oracle => Ok(Format::OracleGeneral),
        CliFormat::Csv => Ok(Format::SyntheticCsv),
    }
}

fn resolve_sizes(cli: &Cli) -> anyhow::Result<Vec<usize>> {
    let mut sizes = Vec::new();
    if let Some(sz) = cli.cache_size {
        sizes.push(sz);
    }
    sizes.extend_from_slice(&cli.cache_sizes);
    if let Some(grid) = &cli.grid {
        let path = std::path::PathBuf::from(format!("grids/{grid}.json"));
        anyhow::bail!(
            "grid '{grid}' not yet populated (expected at {}); pass --cache-size or --cache-sizes for now",
            path.display()
        );
    }
    if sizes.is_empty() {
        anyhow::bail!("no cache size specified: pass --cache-size, --cache-sizes, or --grid");
    }
    Ok(sizes)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .context("rayon thread pool init")?;
    }

    let fmt = resolve_format(&cli.format, &cli.trace)?;
    let sizes = resolve_sizes(&cli)?;

    let mut reports = Vec::new();
    for pol in &cli.policy {
        let kind: PolicyKind = pol.clone().into();
        for &size in &sizes {
            let iter = trace::open(&cli.trace, fmt)?;
            let run = cachesim::metrics::run_single(policy::make(kind, size), iter)?;
            reports.push(cachesim::report::build(&run, None));
        }
    }

    if cli.real.is_some() {
        eprintln!("[cachesim] --real is wired but HRC-MAE pairing lands with the Mattson pass; reporting fake-side only for now");
    }

    let json = serde_json::to_string_pretty(&reports)?;
    if cli.out == "-" {
        println!("{json}");
    } else {
        std::fs::write(&cli.out, json)?;
    }
    Ok(())
}
