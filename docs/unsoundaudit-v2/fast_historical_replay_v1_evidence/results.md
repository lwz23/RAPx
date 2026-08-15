# RAPx 快速历史源码回放 v1

该表是历史源码诊断，不是 recall、precision 或生态 coverage 结论。

| case | expected_pattern | repo_issue | detached_revision | control | rap | p1 | p2 | p3 | p4 | p5 | p6 | result_dir | outcome |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | P1 | servo/rust-smallvec #343 | `source_unavailable` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-01 | source_unavailable |
| 2 | P1 | GitoxideLabs/gitoxide #1460 | `a807dd1ffb05efd177700d065095249e6c4b3c68` | ok | fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-02 | analyzer_failed |
| 3 | P2 | rust-lang/libc #3560 | `96ed9c65448a9f9f5c9095bf537435ffecff627a` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-03 | lockfile_missing |
| 4 | P3 | jeromefroe/lru-rs #73 | `53f53f6e762405be5a73ca87becb25d308e9d34b` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-04 | lockfile_missing |
| 5 | P3 | hyperium/http #639 | `source_unavailable` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-05 | source_unavailable |
| 6 | P4 | fitzgen/bumpalo #164 | `source_unavailable` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-06 | source_unavailable |
| 7 | P4 | rust-random/rand #779 | `9828cdf0201f816e4e4bff07834d849a10be80e9` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-07 | lockfile_missing |
| 8 | P5.1 | rust-random/rand #1158 | `64ac013b2a9754dc114559958eaa72762ce6b582` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-08 | lockfile_missing |
| 9 | P6 | nix-rust/nix #1819 | `12fb35434f04dccf941c3e1e52fb98df6cea41e6` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-09 | lockfile_missing |
| 10 | P6 | diesel-rs/diesel #4223 | `08911c8420f4e2ff732a3b263a5a4f3df0f745d2` | fail | skipped_control_fail | 0 | 0 | 0 | 0 | 0 | 0 | cases/case-10 | lockfile_missing |
