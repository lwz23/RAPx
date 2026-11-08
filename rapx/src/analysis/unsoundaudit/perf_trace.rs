use serde_json::{Map, Number, Value};
use std::collections::BTreeSet;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TRACE_ENV: &str = "UNSOUND_SCANNER_PERF_TRACE";
const TRACE_SCHEMA: &str = "unsoundaudit-v2-perf-trace-v2";

pub(super) struct PerfTrace {
    writer: Option<BufWriter<File>>,
    started: Instant,
    active_phases: BTreeSet<String>,
}

impl PerfTrace {
    pub(super) fn is_enabled(&self) -> bool {
        self.writer.is_some()
    }

    pub(super) fn from_environment() -> Result<Self, String> {
        match env::var_os(TRACE_ENV) {
            Some(path) => Self::create(PathBuf::from(path), Instant::now()),
            None => Ok(Self::disabled(Instant::now())),
        }
    }

    fn disabled(started: Instant) -> Self {
        Self {
            writer: None,
            started,
            active_phases: BTreeSet::new(),
        }
    }

    fn create(path: impl AsRef<Path>, started: Instant) -> Result<Self, String> {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(format!("{TRACE_ENV} must name a fresh trace file"));
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                format!(
                    "failed to create fresh UnsoundAudit performance trace '{}': {error}",
                    path.display()
                )
            })?;
        Ok(Self {
            writer: Some(BufWriter::new(file)),
            started,
            active_phases: BTreeSet::new(),
        })
    }

    pub(super) fn event<const N: usize>(
        &mut self,
        phase: &str,
        event: &str,
        counts: [(&str, u64); N],
    ) -> Result<(), String> {
        let Some(writer) = &mut self.writer else {
            return Ok(());
        };
        validate_name("phase", phase)?;
        if !matches!(event, "begin" | "end" | "checkpoint") {
            return Err(format!(
                "UnsoundAudit performance trace event must be 'begin', 'end', or 'checkpoint', got '{event}'"
            ));
        }
        let mut count_values = Map::new();
        for (name, value) in counts {
            validate_name("count", name)?;
            if count_values
                .insert(name.to_string(), Value::Number(Number::from(value)))
                .is_some()
            {
                return Err(format!(
                    "duplicate UnsoundAudit performance trace count '{name}'"
                ));
            }
        }
        match event {
            "begin" if !self.active_phases.insert(phase.to_string()) => {
                return Err(format!(
                    "UnsoundAudit performance trace phase '{phase}' is already active"
                ));
            }
            "end" if !self.active_phases.remove(phase) => {
                return Err(format!(
                    "UnsoundAudit performance trace phase '{phase}' is not active"
                ));
            }
            _ => {}
        }
        let elapsed_us = u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let mut record = Map::new();
        record.insert(
            "schema".to_string(),
            Value::String(TRACE_SCHEMA.to_string()),
        );
        record.insert("phase".to_string(), Value::String(phase.to_string()));
        record.insert("event".to_string(), Value::String(event.to_string()));
        record.insert(
            "elapsed_us".to_string(),
            Value::Number(Number::from(elapsed_us)),
        );
        record.insert("counts".to_string(), Value::Object(count_values));
        serde_json::to_writer(&mut *writer, &Value::Object(record)).map_err(|error| {
            format!("failed to serialize UnsoundAudit performance trace: {error}")
        })?;
        writer
            .write_all(b"\n")
            .and_then(|()| writer.flush())
            .map_err(|error| format!("failed to flush UnsoundAudit performance trace: {error}"))
    }

    pub(super) fn finish(mut self) -> Result<(), String> {
        if !self.active_phases.is_empty() {
            return Err(format!(
                "UnsoundAudit performance trace has unfinished phases: {:?}",
                self.active_phases
            ));
        }
        if let Some(writer) = &mut self.writer {
            writer.flush().map_err(|error| {
                format!("failed to finish UnsoundAudit performance trace: {error}")
            })?;
        }
        Ok(())
    }
}

fn validate_name(kind: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(format!(
            "UnsoundAudit performance trace {kind} must match [a-z0-9_]+, got '{value}'"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PerfTrace, TRACE_SCHEMA};
    use serde_json::Value;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    static TRACE_ID: AtomicU64 = AtomicU64::new(0);

    fn fresh_path(name: &str) -> std::path::PathBuf {
        let id = TRACE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rapx-unsoundaudit-{name}-{}-{id}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn trace_records_closed_jsonl_and_flushes_each_event() {
        let path = fresh_path("closed-trace");
        let mut trace = PerfTrace::create(&path, Instant::now()).unwrap();
        trace.event("analysis", "begin", []).unwrap();
        trace
            .event("analysis", "end", [("success", 1), ("finding_count", 2)])
            .unwrap();
        trace.finish().unwrap();

        let text = fs::read_to_string(&path).unwrap();
        let records = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        for record in &records {
            assert_eq!(
                record
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>(),
                ["counts", "elapsed_us", "event", "phase", "schema"]
            );
            assert_eq!(record["schema"], TRACE_SCHEMA);
            assert!(record["elapsed_us"].as_u64().is_some());
        }
        assert_eq!(records[0]["event"], "begin");
        assert_eq!(records[1]["counts"]["finding_count"], 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn disabled_trace_writes_nothing_and_accepts_balanced_events() {
        let mut trace = PerfTrace::disabled(Instant::now());
        assert!(!trace.is_enabled());
        trace.event("analysis", "begin", []).unwrap();
        trace
            .event("analysis_progress", "checkpoint", [("ordinal", 1)])
            .unwrap();
        trace.event("analysis", "end", [("success", 1)]).unwrap();
        trace.finish().unwrap();
    }

    #[test]
    fn checkpoint_flushes_without_changing_active_phase_accounting() {
        let path = fresh_path("checkpoint-trace");
        assert_eq!(TRACE_SCHEMA, "unsoundaudit-v2-perf-trace-v2");
        let mut trace = PerfTrace::create(&path, Instant::now()).unwrap();
        assert!(trace.is_enabled());
        trace.event("analysis", "begin", []).unwrap();
        trace
            .event("analysis_progress", "checkpoint", [("ordinal", 1)])
            .unwrap();
        trace.event("analysis", "end", []).unwrap();
        trace.finish().unwrap();

        let records = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 3);
        assert_eq!(records[1]["event"], "checkpoint");
        assert_eq!(records[1]["counts"]["ordinal"], 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn trace_refuses_overwrite_invalid_names_and_duplicate_counts() {
        let path = fresh_path("fail-closed-trace");
        let mut trace = PerfTrace::create(&path, Instant::now()).unwrap();
        assert!(PerfTrace::create(&path, Instant::now()).is_err());
        assert!(trace.event("Bad Phase", "begin", []).is_err());
        assert!(trace.event("analysis", "middle", []).is_err());
        assert!(trace
            .event("analysis", "end", [("same", 1), ("same", 2)])
            .is_err());
        trace.event("analysis", "begin", []).unwrap();
        assert!(trace.event("analysis", "begin", []).is_err());
        trace.event("analysis", "end", []).unwrap();
        assert!(trace.event("analysis", "end", []).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn trace_finish_rejects_an_unfinished_phase_after_flushing_a_valid_prefix() {
        let path = fresh_path("unfinished-trace");
        let mut trace = PerfTrace::create(&path, Instant::now()).unwrap();
        trace.event("analysis", "begin", []).unwrap();
        assert!(trace.finish().is_err());
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);
        fs::remove_file(path).unwrap();
    }
}
