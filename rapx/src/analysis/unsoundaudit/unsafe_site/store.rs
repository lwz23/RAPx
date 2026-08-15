use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use once_cell::sync::Lazy;

use super::binding::MirEventAnchor;
use super::{CoverageFailure, FunctionKey, OwnerCensus};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CensusSnapshot {
    pub generation: u64,
    pub owners: BTreeMap<FunctionKey, OwnerCensus>,
    pub built_events: BTreeMap<FunctionKey, Vec<MirEventAnchor>>,
    pub failures: BTreeSet<CoverageFailure>,
}

impl CensusSnapshot {
    fn empty(generation: u64) -> Self {
        Self {
            generation,
            owners: BTreeMap::new(),
            built_events: BTreeMap::new(),
            failures: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct CensusStore {
    state: Mutex<Option<CensusSnapshot>>,
}

impl CensusStore {
    pub fn begin(&self, generation: u64) -> Result<(), String> {
        if generation == 0 {
            return Err("census generation zero is reserved".to_string());
        }
        let mut state = self.lock()?;
        *state = Some(CensusSnapshot::empty(generation));
        Ok(())
    }

    pub fn record_owner(&self, generation: u64, census: OwnerCensus) -> Result<(), String> {
        census.validate()?;
        let mut state = self.lock()?;
        let snapshot = matching_snapshot(&mut state, generation)?;
        if snapshot.owners.contains_key(&census.owner) {
            return Err(format!(
                "owner census already recorded for {}",
                census.owner.0
            ));
        }
        snapshot.owners.insert(census.owner.clone(), census);
        Ok(())
    }

    pub fn record_failure(&self, generation: u64, failure: CoverageFailure) -> Result<(), String> {
        let mut state = self.lock()?;
        let snapshot = matching_snapshot(&mut state, generation)?;
        if !snapshot.failures.insert(failure) {
            return Err("duplicate census coverage failure".to_string());
        }
        Ok(())
    }

    pub fn record_built_events(
        &self,
        generation: u64,
        owner: FunctionKey,
        events: Vec<MirEventAnchor>,
    ) -> Result<(), String> {
        if events.iter().any(|event| event.owner != owner) {
            return Err("built MIR event owner differs from its store key".to_string());
        }
        let mut state = self.lock()?;
        let snapshot = matching_snapshot(&mut state, generation)?;
        if snapshot.built_events.insert(owner, events).is_some() {
            return Err("built MIR events already recorded for owner".to_string());
        }
        Ok(())
    }

    pub fn contains_built_events(
        &self,
        generation: u64,
        owner: &FunctionKey,
    ) -> Result<bool, String> {
        let state = self.lock()?;
        let snapshot = state
            .as_ref()
            .ok_or_else(|| "census invocation has not begun".to_string())?;
        if snapshot.generation != generation {
            return Err(format!(
                "census generation mismatch: active {}, requested {generation}",
                snapshot.generation
            ));
        }
        Ok(snapshot.built_events.contains_key(owner))
    }

    pub fn contains_owner(&self, generation: u64, owner: &FunctionKey) -> Result<bool, String> {
        let state = self.lock()?;
        let snapshot = state
            .as_ref()
            .ok_or_else(|| "census invocation has not begun".to_string())?;
        if snapshot.generation != generation {
            return Err(format!(
                "census generation mismatch: active {}, requested {generation}",
                snapshot.generation
            ));
        }
        Ok(snapshot.owners.contains_key(owner))
    }

    pub fn snapshot(&self, generation: u64) -> Result<CensusSnapshot, String> {
        let state = self.lock()?;
        let snapshot = state
            .as_ref()
            .ok_or_else(|| "census invocation has not begun".to_string())?;
        if snapshot.generation != generation {
            return Err(format!(
                "census generation mismatch: active {}, requested {generation}",
                snapshot.generation
            ));
        }
        Ok(snapshot.clone())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<CensusSnapshot>>, String> {
        self.state
            .lock()
            .map_err(|_| "census store mutex poisoned".to_string())
    }
}

fn matching_snapshot(
    state: &mut Option<CensusSnapshot>,
    generation: u64,
) -> Result<&mut CensusSnapshot, String> {
    let snapshot = state
        .as_mut()
        .ok_or_else(|| "census invocation has not begun".to_string())?;
    if snapshot.generation != generation {
        return Err(format!(
            "census generation mismatch: active {}, requested {generation}",
            snapshot.generation
        ));
    }
    Ok(snapshot)
}

static GLOBAL_STORE: Lazy<CensusStore> = Lazy::new(CensusStore::default);
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
static ACTIVE_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn begin_global_invocation() -> Result<u64, String> {
    let generation = NEXT_GENERATION.fetch_add(1, Ordering::SeqCst);
    if generation == 0 || generation == u64::MAX {
        return Err("census invocation generation exhausted".to_string());
    }
    GLOBAL_STORE.begin(generation)?;
    ACTIVE_GENERATION.store(generation, Ordering::SeqCst);
    Ok(generation)
}

pub fn active_generation() -> Result<u64, String> {
    let generation = ACTIVE_GENERATION.load(Ordering::SeqCst);
    if generation == 0 {
        return Err("census invocation has not begun".to_string());
    }
    Ok(generation)
}

pub fn global_store() -> &'static CensusStore {
    &GLOBAL_STORE
}
