//! Opt-in, synchronized wall-clock attribution for the real training CLI.
//! CNOT103_STAGE_PROFILE=1 writes one JSON record to stderr. Nested stages
//! report exclusive time. Throughput comparisons should use unprofiled runs.
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Trace {
    device: Option<usize>,
    devices: Vec<usize>,
    stack: Vec<Duration>,
    stages: BTreeMap<&'static str, (u64, Duration, Duration)>,
    chunk_limit: Option<u64>,
    chunk_start: usize,
    chunk_count: u64,
    chunk_seconds: Vec<f64>,
}
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TRACE: RefCell<Trace> = RefCell::new(Trace::default());
}

pub(crate) struct Session(Option<Instant>);

impl Session {
    pub(crate) fn start() -> Self {
        if std::env::var_os("CNOT103_STAGE_PROFILE").as_deref() != Some(std::ffi::OsStr::new("1")) {
            return Self(None);
        }
        let chunk_limit = std::env::var("CNOT103_PROFILE_CHUNKS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0);
        let chunk_start = std::env::var("CNOT103_PROFILE_CHUNK_START")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        TRACE.with(|t| {
            *t.borrow_mut() = Trace {
                chunk_limit,
                chunk_start,
                ..Trace::default()
            }
        });
        ACTIVE.with(|a| a.set(true));
        Self(Some(Instant::now()))
    }

    pub(crate) fn check_chunk_probe(&self) -> Result<(), String> {
        if let Some(raw) = std::env::var_os("CNOT103_PROFILE_CHUNKS") {
            if self.0.is_none()
                || raw
                    .to_str()
                    .and_then(|v| v.parse::<u64>().ok())
                    .is_none_or(|n| n == 0)
            {
                return Err(
                    "CNOT103_PROFILE_CHUNKS requires a positive count and CNOT103_STAGE_PROFILE=1"
                        .into(),
                );
            }
        }
        if let Some(raw) = std::env::var_os("CNOT103_PROFILE_CHUNK_START") {
            if self.0.is_none()
                || !self.probing_candidates()
                || raw.to_str().and_then(|s| s.parse::<usize>().ok()).is_none()
            {
                return Err(
                    "CNOT103_PROFILE_CHUNK_START requires a nonnegative index and an enabled candidate chunk probe"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn device(&self, index: usize) {
        if self.0.is_some() {
            TRACE.with(|t| {
                let mut trace = t.borrow_mut();
                trace.device = Some(index);
                trace.devices = vec![index];
            });
        }
    }

    /// Record all CUDA devices whose queued work must finish at stage boundaries.
    pub(crate) fn devices(&self, indices: &[usize]) {
        if self.0.is_some() {
            TRACE.with(|t| {
                let mut trace = t.borrow_mut();
                trace.device = indices.first().copied();
                trace.devices = indices.to_vec();
            });
        }
    }

    pub(crate) fn probing_candidates(&self) -> bool {
        self.0.is_some() && TRACE.with(|trace| trace.borrow().chunk_limit.is_some())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            synchronize_devices();
            ACTIVE.with(|a| a.set(false));
            TRACE.with(|trace| {
                let trace = trace.take();
                let chunk_probe = trace.chunk_limit.map(|limit| {
                    serde_json::json!({
                        "limit": limit, "completed": trace.chunk_count
                    })
                });
                let wall = start.elapsed();
                let measured: Duration = trace.stages.values().map(|(_, time, _)| *time).sum();
                let stages: BTreeMap<_, _> = trace
                    .stages
                    .into_iter()
                    .map(|(name, (calls, elapsed, inclusive))| {
                        let ms = elapsed.as_secs_f64() * 1000.0;
                        (
                            name,
                            serde_json::json!({ "calls": calls, "exclusive_ms": ms,
                                               "inclusive_ms": inclusive.as_secs_f64() * 1000.0 }),
                        )
                    })
                    .collect();
                let summary = serde_json::json!({
                    "schema": "cnot103_stage_profile_v1",
                    "wall_ms": wall.as_secs_f64() * 1000.0,
                    "unattributed_ms": wall.saturating_sub(measured).as_secs_f64() * 1000.0,
                    "candidate_chunk_probe": chunk_probe,
                    "candidate_chunk_start_index": trace.chunk_start,
                    "candidate_chunk_seconds": trace.chunk_seconds,
                    "stages": stages
                });
                eprintln!("{summary}");
            });
        }
    }
}

/// In profiling-only mode, discard earlier candidate chunks before timing.
/// A candidate probe always aborts before gradient computation or an update.
pub(crate) fn skip_candidate_chunk(chunk: usize) -> bool {
    ACTIVE.with(Cell::get)
        && TRACE.with(|trace| {
            let trace = trace.borrow();
            trace.chunk_limit.is_some() && chunk < trace.chunk_start
        })
}

#[cfg(test)]
mod candidate_chunk_offset_tests {
    use super::*;

    #[test]
    fn offset_skips_only_in_active_candidate_probes() {
        TRACE.with(|trace| {
            *trace.borrow_mut() = Trace {
                chunk_limit: Some(3),
                chunk_start: 5,
                ..Trace::default()
            };
        });
        ACTIVE.with(|active| active.set(false));
        assert!(!skip_candidate_chunk(0));
        ACTIVE.with(|active| active.set(true));
        assert!(skip_candidate_chunk(0));
        assert!(skip_candidate_chunk(4));
        assert!(!skip_candidate_chunk(5));
        assert!(!skip_candidate_chunk(6));
        TRACE.with(|trace| trace.borrow_mut().chunk_limit = None);
        assert!(!skip_candidate_chunk(0));
        ACTIVE.with(|active| active.set(false));
        TRACE.with(|trace| *trace.borrow_mut() = Trace::default());
    }
}

fn synchronize_devices() {
    let devices = TRACE.with(|trace| trace.borrow().devices.clone());
    for index in devices {
        tch::Cuda::synchronize(index as i64);
    }
}

/// Start an independent timing sample only for a bounded candidate probe.
pub(crate) fn candidate_chunk_start() -> Option<Instant> {
    if !ACTIVE.with(Cell::get) {
        return None;
    }
    TRACE.with(|trace| {
        let trace = trace.borrow();
        trace.chunk_limit?;
        tch::Cuda::synchronize(trace.device? as i64);
        Some(Instant::now())
    })
}

/// Stop a requested timing probe before any gradient computation or update.
pub(crate) fn candidate_chunk_complete(start: Option<Instant>) -> Result<(), String> {
    if !ACTIVE.with(Cell::get) {
        return Ok(());
    }
    TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        if let Some(limit) = trace.chunk_limit {
            let device = trace.device.ok_or("candidate probe has no CUDA device")?;
            tch::Cuda::synchronize(device as i64);
            trace.chunk_seconds.push(
                start
                    .ok_or("candidate probe timer did not start")?
                    .elapsed()
                    .as_secs_f64(),
            );
            trace.chunk_count += 1;
            if trace.chunk_count == limit {
                return Err(format!(
                    "requested profiling stop after {limit} candidate chunks"
                ));
            }
        }
        Ok(())
    })
}

pub(crate) fn finish_candidate_probe() -> Result<(), String> {
    if ACTIVE.with(Cell::get) && TRACE.with(|t| t.borrow().chunk_limit.is_some()) {
        return Err("requested candidate chunk count exceeds this record's neighbourhood".into());
    }
    Ok(())
}

pub(crate) struct Stage(Option<(&'static str, Instant)>);

fn progress(event: &str, name: &str) {
    if std::env::var_os("CNOT103_STAGE_PROFILE_PROGRESS").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        eprintln!("CNOT103_STAGE {event} {name}");
    }
}

pub(crate) fn stage(name: &'static str) -> Stage {
    if !ACTIVE.with(Cell::get) {
        return Stage(None);
    }
    synchronize_devices();
    TRACE.with(|t| t.borrow_mut().stack.push(Duration::ZERO));
    progress("begin", name);
    Stage(Some((name, Instant::now())))
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some((name, start)) = self.0 {
            synchronize_devices();
            TRACE.with(|t| {
                let mut trace = t.borrow_mut();
                let elapsed = start.elapsed();
                let children = trace.stack.pop().expect("profiling scope");
                if let Some(parent) = trace.stack.last_mut() {
                    *parent += elapsed;
                }
                let (calls, total, inclusive) = trace.stages.entry(name).or_default();
                *calls += 1;
                *total += elapsed.saturating_sub(children);
                *inclusive += elapsed;
            });
            progress("end", name);
        }
    }
}

pub(crate) fn measure<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let _stage = stage(name);
    f()
}
