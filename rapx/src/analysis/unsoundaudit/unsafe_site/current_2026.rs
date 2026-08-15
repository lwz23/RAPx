use std::collections::BTreeSet;

use super::{AdapterManifest, UnsafeSiteKind};

pub const RUSTC_COMMIT: &str = "ba28ff76f353a722f31c4f3dd2ac4e437d36411b";
const CHECKER_SHA256: &str = "25b4b8e22ff7d561dda2a9abf429ab5bef035d246ec0200b9b887f7713fca346";

pub fn manifest() -> AdapterManifest {
    AdapterManifest {
        adapter_id: "rustc-nightly-2026-08-14-ba28ff".to_string(),
        rustc_commit: RUSTC_COMMIT.to_string(),
        checker_sha256: CHECKER_SHA256.to_string(),
        emitted_kinds: BTreeSet::from([
            UnsafeSiteKind::UnsafeCall,
            UnsafeSiteKind::InlineAssembly,
            UnsafeSiteKind::UnsafeFieldInitialization,
            UnsafeSiteKind::UnsafeFieldUse,
            UnsafeSiteKind::MutableStaticAccess,
            UnsafeSiteKind::ExternStaticAccess,
            UnsafeSiteKind::RawPointerDereference,
            UnsafeSiteKind::UnionFieldAccess,
            UnsafeSiteKind::TargetFeatureCall,
            UnsafeSiteKind::UnsafeBinderCast,
            UnsafeSiteKind::ExplicitDropCall,
        ]),
    }
}
