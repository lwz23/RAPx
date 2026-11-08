# lwz23 有效 Issue RAP v3 扫描进度

扫描器：当前工作树 RAP v3 `rapx-perf`，toolchain `nightly-2026-08-14` / rustc `ba28ff76`。
源码：上报前默认分支 revision。waterfox 未扫。

- 已完成仓库：5 / 124
- 仓库状态：{'miss': 3, 'hit': 1, 'control_failed': 1}
- 已完成扫描中 cover 住的上报 Issue：1
- 扫描完成但未匹配上报标题的 Issue：4

## 已 cover
- [Andy-Python-Programmer/aero](https://github.com/Andy-Python-Programmer/aero/issues/120) Unsoundness in Utsname::name and other function
  - `P1` `Utsname::machine` → `core::str::from_utf8_unchecked` `src/aero_syscall/src/lib.rs:254`
  - `P1` `Utsname::version` → `core::str::from_utf8_unchecked` `src/aero_syscall/src/lib.rs:250`
  - `P1` `Utsname::nodename` → `core::str::from_utf8_unchecked` `src/aero_syscall/src/lib.rs:242`
  - `P1` `Utsname::name` → `core::str::from_utf8_unchecked` `src/aero_syscall/src/lib.rs:238`
  - `P1` `Utsname::release` → `core::str::from_utf8_unchecked` `src/aero_syscall/src/lib.rs:246`

## 扫描完成但未匹配
- Amanieu/atomic-rs: Maybe Multiple unsoundness problems（active=0）
- Andy-Python-Programmer/aero: Maybe unsound in some as_ptr usage（active=6）
- CosmWasm/cosmwasm: Maybe unsound in as_bytes（active=0）
- Eugeny/russh: Potential unsoundness in memcpy（active=4）

## 尚未得到检测结论
- DioxusLabs/blitz: Unsoundness problem in Node::tree status=control_failed

