use std::collections::BTreeSet;

use super::{AdapterManifest, UnsafeSiteKind};

pub const RUSTC_COMMIT: &str = "1bc403daadbebb553ccc211a0a8eebb73989665f";
const CHECKER_SHA256: &str = "bde4d27635e54d8ca0582ef8decd204c3089591e5c540322891d7aaf1a182023";

pub fn manifest() -> AdapterManifest {
    AdapterManifest {
        adapter_id: "rustc-nightly-2024-10-12-1bc403".to_string(),
        rustc_commit: RUSTC_COMMIT.to_string(),
        checker_sha256: CHECKER_SHA256.to_string(),
        emitted_kinds: BTreeSet::from([
            UnsafeSiteKind::UnsafeCall,
            UnsafeSiteKind::InlineAssembly,
            UnsafeSiteKind::RestrictedValidityInitialization,
            UnsafeSiteKind::MutableStaticAccess,
            UnsafeSiteKind::ExternStaticAccess,
            UnsafeSiteKind::RawPointerDereference,
            UnsafeSiteKind::UnionFieldAccess,
            UnsafeSiteKind::LayoutConstrainedMutation,
            UnsafeSiteKind::LayoutConstrainedBorrow,
            UnsafeSiteKind::TargetFeatureCall,
        ]),
    }
}
