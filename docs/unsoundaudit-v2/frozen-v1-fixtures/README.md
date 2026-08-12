# UnsoundAudit pattern regression fixtures

Every child directory is an independent Cargo package.  Run each package in
isolation, without a timeout, and with a fresh target/cache and receipt
directory. Point RAP's per-unit output directory at:

```text
<receipts-root>/<case-id>/*.json
```

Do not use the legacy single `UNSOUND_SCANNER_RAP_JSON_OUT` artifact as the
test input.  Once all cases have produced `rap-unit-v1` receipts, validate
them with:

```text
python3 scripts/check_unsoundaudit_pattern_receipts.py \
  --receipts-root <receipts-root>
```

`fixture.json` is the hand-written oracle for each package.  Positive cases
expect one finding for exactly one pattern.  Negative cases expect zero for
all four patterns.  `pattern2_display_negative` and
`pattern2_write_negative` specifically protect ordinary formatting code from
the known Pattern2 false-positive risk.

The checker is deliberately offline: it only reads fixture metadata and
already-produced receipts. It never launches Cargo, rustc, or RAP.
