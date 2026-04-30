use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;

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
    Fifo,
    Sieve,
    Slru,
    Car,
}

impl From<CliPolicy> for PolicyKind {
    fn from(p: CliPolicy) -> PolicyKind {
        match p {
            CliPolicy::Lru => PolicyKind::Lru,
            CliPolicy::Arc => PolicyKind::Arc,
            CliPolicy::Fifo => PolicyKind::Fifo,
            CliPolicy::Sieve => PolicyKind::Sieve,
            CliPolicy::Slru => PolicyKind::Slru,
            CliPolicy::Car => PolicyKind::Car,
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

    // Load the trace once into memory, time-sort by ts (so block-
    // concatenated multi-stream CSVs interleave naturally), and reduce to
    // a Vec<u64> cache-key stream. Cache key = (stream_id, obj_id) →
    // unique 64-bit id assigned in first-seen order, so streams with the
    // same `obj_id` namespace stay disjoint.
    let t0 = std::time::Instant::now();
    let mut accesses: Vec<cachesim::trace::Access> = trace::open(&cli.trace, fmt)?
        .collect::<anyhow::Result<Vec<_>>>()?;
    accesses.sort_by_key(|a| a.ts);
    let mut keys: std::collections::HashMap<(u32, u64), u64> =
        std::collections::HashMap::with_capacity(accesses.len() / 2 + 1);
    let ids: Vec<u64> = accesses
        .iter()
        .map(|a| {
            let next = keys.len() as u64;
            *keys.entry((a.stream_id, a.obj_id)).or_insert(next)
        })
        .collect();
    eprintln!(
        "[cachesim] loaded {} accesses ({} unique cache keys, {} streams) from {} in {:.1}s",
        ids.len(),
        keys.len(),
        accesses.iter().map(|a| a.stream_id).collect::<std::collections::BTreeSet<_>>().len(),
        cli.trace.display(),
        t0.elapsed().as_secs_f64()
    );
    drop(accesses);

    let kinds: Vec<PolicyKind> = cli.policy.iter().cloned().map(Into::into).collect();
    let jobs: Vec<(PolicyKind, usize)> = kinds
        .iter()
        .flat_map(|&k| sizes.iter().map(move |&s| (k, s)))
        .collect();

    let reports: Vec<_> = jobs
        .par_iter()
        .map(|&(kind, size)| -> anyhow::Result<_> {
            let t = std::time::Instant::now();
            let iter = ids.iter().map(|&obj_id| {
                Ok(cachesim::trace::Access {
                    obj_id,
                    ..Default::default()
                })
            });
            let run = cachesim::metrics::run_single(policy::make(kind, size), iter)?;
            eprintln!(
                "[cachesim] {} cap={} done in {:.1}s (miss_ratio={:.4})",
                run.policy,
                size,
                t.elapsed().as_secs_f64(),
                run.per_cache_size
                    .first()
                    .map(|p| p.miss_ratio)
                    .unwrap_or(f64::NAN),
            );
            Ok(cachesim::report::build(&run, None))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

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
