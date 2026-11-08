# lwz23 有效 Issue 扫描 Roster v1

本文件冻结 RAP 回放目标。在发布 v2 之前，只扫这里的仓库和 Issue。

## 冻结口径

- 作者：`lwz23`（GitHub user id `160561672`）
- 窗口：`[2024-11-01T00:00:00Z, 2026-01-01T00:00:00Z)`
- 有效：维护者未明确否定，且作者未明确撤回
- 权威来源：Task4AL v2 `issue_outcomes.jsonl` + Task4AM `valid_issues_index.csv`

## 计数

| 项目 | 数量 |
|---|---:|
| 上报 Issue | 160 |
| 非 unsound 排除 | 2 |
| in-scope | 158 |
| **有效 Issue** | **148** |
| 明确反驳或撤回 | 10 |
| 其中可扫 Issue（软件仓） | 142 |
| advisory-db 有效 Issue（不单独扫） | 6 |
| **扫描仓库** | **125** |
| 维护者认可并修复 | 27 |

先前口头说的 132 仓，是 158 条 in-scope Issue 去掉 `rustsec/advisory-db` 后的宿主仓。
去掉 10 条明确否定后，有 7 个仓不再剩有效 Issue，扫描仓定为 **125**。

本轮仍未冻结：脆弱 revision、package/target、lockfile 策略。先定仓库和 Issue 链接。

## 扫描仓库与 Issue 链接

### [Amanieu/atomic-rs](https://github.com/Amanieu/atomic-rs)

- clone: `https://github.com/Amanieu/atomic-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [Amanieu/atomic-rs#46](https://github.com/Amanieu/atomic-rs/issues/46) Maybe Multiple unsoundness problems

### [Andy-Python-Programmer/aero](https://github.com/Andy-Python-Programmer/aero)

- clone: `https://github.com/Andy-Python-Programmer/aero.git`
- Issue 数：2；accepted-and-fixed：0
- [Andy-Python-Programmer/aero#120](https://github.com/Andy-Python-Programmer/aero/issues/120) Unsoundness in Utsname::name and other function
- [Andy-Python-Programmer/aero#121](https://github.com/Andy-Python-Programmer/aero/issues/121) Maybe unsound in some as_ptr usage

### [BrowserWorks/waterfox](https://github.com/BrowserWorks/waterfox)

- clone: `https://github.com/BrowserWorks/waterfox.git`
- Issue 数：1；accepted-and-fixed：0
- [BrowserWorks/waterfox#3980](https://github.com/BrowserWorks/waterfox/issues/3980) [Unsoundness] Safe function peek_from_slice causes UB via malicious Peek implementation

### [CosmWasm/cosmwasm](https://github.com/CosmWasm/cosmwasm)

- clone: `https://github.com/CosmWasm/cosmwasm.git`
- Issue 数：1；accepted-and-fixed：1
- [CosmWasm/cosmwasm#2305](https://github.com/CosmWasm/cosmwasm/issues/2305) Maybe unsound in as_bytes · 已认可修复

### [DioxusLabs/blitz](https://github.com/DioxusLabs/blitz)

- clone: `https://github.com/DioxusLabs/blitz.git`
- Issue 数：1；accepted-and-fixed：1
- [DioxusLabs/blitz#169](https://github.com/DioxusLabs/blitz/issues/169) Unsoundness problem in Node::tree · 已认可修复

### [Eugeny/russh](https://github.com/Eugeny/russh)

- clone: `https://github.com/Eugeny/russh.git`
- Issue 数：1；accepted-and-fixed：1
- [Eugeny/russh#394](https://github.com/Eugeny/russh/issues/394) Potential unsoundness in memcpy · 已认可修复

### [Indra-db/Flecs-Rust](https://github.com/Indra-db/Flecs-Rust)

- clone: `https://github.com/Indra-db/Flecs-Rust.git`
- Issue 数：1；accepted-and-fixed：0
- [Indra-db/Flecs-Rust#210](https://github.com/Indra-db/Flecs-Rust/issues/210) Maybe unsound in Cursor::new

### [IsaacWoods/poplar](https://github.com/IsaacWoods/poplar)

- clone: `https://github.com/IsaacWoods/poplar.git`
- Issue 数：1；accepted-and-fixed：0
- [IsaacWoods/poplar#117](https://github.com/IsaacWoods/poplar/issues/117) Maybe unsound in dma::write

### [Kudaes/DInvoke_rs](https://github.com/Kudaes/DInvoke_rs)

- clone: `https://github.com/Kudaes/DInvoke_rs.git`
- Issue 数：1；accepted-and-fixed：0
- [Kudaes/DInvoke_rs#10](https://github.com/Kudaes/DInvoke_rs/issues/10) Unsoundness in fn clean_dos_header

### [Kudaes/Elevator](https://github.com/Kudaes/Elevator)

- clone: `https://github.com/Kudaes/Elevator.git`
- Issue 数：1；accepted-and-fixed：0
- [Kudaes/Elevator#9](https://github.com/Kudaes/Elevator/issues/9) Maybe unsound in get_pe_metadata

### [Lightprotocol/light-protocol](https://github.com/Lightprotocol/light-protocol)

- clone: `https://github.com/Lightprotocol/light-protocol.git`
- Issue 数：1；accepted-and-fixed：1
- [Lightprotocol/light-protocol#1382](https://github.com/Lightprotocol/light-protocol/issues/1382) Maybe unsound in ConcurrentMerkleTree · 已认可修复

### [MarcusRainbow/QuantMath](https://github.com/MarcusRainbow/QuantMath)

- clone: `https://github.com/MarcusRainbow/QuantMath.git`
- Issue 数：1；accepted-and-fixed：0
- [MarcusRainbow/QuantMath#54](https://github.com/MarcusRainbow/QuantMath/issues/54) Multiple unsound issue

### [MegEngine/MegFlow](https://github.com/MegEngine/MegFlow)

- clone: `https://github.com/MegEngine/MegFlow.git`
- Issue 数：1；accepted-and-fixed：0
- [MegEngine/MegFlow#87](https://github.com/MegEngine/MegFlow/issues/87) Maybe unsound in store

### [Mufanc/z-loader](https://github.com/Mufanc/z-loader)

- clone: `https://github.com/Mufanc/z-loader.git`
- Issue 数：1；accepted-and-fixed：0
- [Mufanc/z-loader#6](https://github.com/Mufanc/z-loader/issues/6) Maybe unsound in SpecializeArgs::env

### [NVSL/Corundum](https://github.com/NVSL/Corundum)

- clone: `https://github.com/NVSL/Corundum.git`
- Issue 数：1；accepted-and-fixed：0
- [NVSL/Corundum#7](https://github.com/NVSL/Corundum/issues/7) Unsoundness in src/vec.rs/to_str and maybe unmaintained

### [NetSys/NetBricks](https://github.com/NetSys/NetBricks)

- clone: `https://github.com/NetSys/NetBricks.git`
- Issue 数：2；accepted-and-fixed：0
- [NetSys/NetBricks#110](https://github.com/NetSys/NetBricks/issues/110) Potential Unsoundness in replace_header
- [NetSys/NetBricks#111](https://github.com/NetSys/NetBricks/issues/111) Potential Unsoundness in remove_from_payload_head and add_to_payload_head

### [Noeda/rllama](https://github.com/Noeda/rllama)

- clone: `https://github.com/Noeda/rllama.git`
- Issue 数：1；accepted-and-fixed：0
- [Noeda/rllama#19](https://github.com/Noeda/rllama/issues/19) Potential Undefined Behavior in data_u16_to_gpu Due to Unsafe Pointer Usage

### [Pagefind/pagefind](https://github.com/Pagefind/pagefind)

- clone: `https://github.com/Pagefind/pagefind.git`
- Issue 数：2；accepted-and-fixed：0
- [Pagefind/pagefind#752](https://github.com/Pagefind/pagefind/issues/752) Maybe unsound in request_filter_indexes and its groups
- [Pagefind/pagefind#985](https://github.com/Pagefind/pagefind/issues/985) [Soundness] Public safe API request_filter_indexes allows dereferencing arbitrary raw pointers

### [PersiaML/PERSIA](https://github.com/PersiaML/PERSIA)

- clone: `https://github.com/PersiaML/PERSIA.git`
- Issue 数：1；accepted-and-fixed：0
- [PersiaML/PERSIA#509](https://github.com/PersiaML/PERSIA/issues/509) Potential Unsoundness in get_shard_by_index Function due to Unchecked Indexing

### [Phala-Network/phala-blockchain](https://github.com/Phala-Network/phala-blockchain)

- clone: `https://github.com/Phala-Network/phala-blockchain.git`
- Issue 数：1；accepted-and-fixed：0
- [Phala-Network/phala-blockchain#1592](https://github.com/Phala-Network/phala-blockchain/issues/1592) Unsoundness in "string_from_c_char"

### [QuarkContainer/Quark](https://github.com/QuarkContainer/Quark)

- clone: `https://github.com/QuarkContainer/Quark.git`
- Issue 数：1；accepted-and-fixed：0
- [QuarkContainer/Quark#1354](https://github.com/QuarkContainer/Quark/issues/1354) Maybe unsound in IOBuff::Buf

### [RWTH-OS/eduOS-rs](https://github.com/RWTH-OS/eduOS-rs)

- clone: `https://github.com/RWTH-OS/eduOS-rs.git`
- Issue 数：1；accepted-and-fixed：1
- [RWTH-OS/eduOS-rs#46](https://github.com/RWTH-OS/eduOS-rs/issues/46) unsoundness in  RomHandle::new · 已认可修复

### [Ralith/hecs](https://github.com/Ralith/hecs)

- clone: `https://github.com/Ralith/hecs.git`
- Issue 数：1；accepted-and-fixed：0
- [Ralith/hecs#385](https://github.com/Ralith/hecs/issues/385) Maybe unsound in 'get_mut'

### [RedisJSON/RedisJSON](https://github.com/RedisJSON/RedisJSON)

- clone: `https://github.com/RedisJSON/RedisJSON.git`
- Issue 数：1；accepted-and-fixed：0
- [RedisJSON/RedisJSON#1305](https://github.com/RedisJSON/RedisJSON/issues/1305) Maybe multiple unsound problem in c_api

### [Ruddle/Fomos](https://github.com/Ruddle/Fomos)

- clone: `https://github.com/Ruddle/Fomos.git`
- Issue 数：1；accepted-and-fixed：0
- [Ruddle/Fomos#5](https://github.com/Ruddle/Fomos/issues/5) Unsoundness in to_bytes

### [SitinCloud/Owlyshield](https://github.com/SitinCloud/Owlyshield)

- clone: `https://github.com/SitinCloud/Owlyshield.git`
- Issue 数：1；accepted-and-fixed：0
- [SitinCloud/Owlyshield#59](https://github.com/SitinCloud/Owlyshield/issues/59) Unsoundness in UnicodeString::as_string_ext

### [Starlight-JS/starlight](https://github.com/Starlight-JS/starlight)

- clone: `https://github.com/Starlight-JS/starlight.git`
- Issue 数：1；accepted-and-fixed：0
- [Starlight-JS/starlight#137](https://github.com/Starlight-JS/starlight/issues/137) Possible unsound problems in crates/starlight/src/vm/environment.rs

### [Teach2Breach/Tempest](https://github.com/Teach2Breach/Tempest)

- clone: `https://github.com/Teach2Breach/Tempest.git`
- Issue 数：1；accepted-and-fixed：0
- [Teach2Breach/Tempest#11](https://github.com/Teach2Breach/Tempest/issues/11) Maybe unsound in get_function_address

### [Traverse-Research/gpu-allocator](https://github.com/Traverse-Research/gpu-allocator)

- clone: `https://github.com/Traverse-Research/gpu-allocator.git`
- Issue 数：1；accepted-and-fixed：1
- [Traverse-Research/gpu-allocator#255](https://github.com/Traverse-Research/gpu-allocator/issues/255) Maybe Unsound in from_winapi_d3d12_resource_desc · 已认可修复

### [Umio-Yasuno/amdgpu_top](https://github.com/Umio-Yasuno/amdgpu_top)

- clone: `https://github.com/Umio-Yasuno/amdgpu_top.git`
- Issue 数：1；accepted-and-fixed：0
- [Umio-Yasuno/amdgpu_top#104](https://github.com/Umio-Yasuno/amdgpu_top/issues/104) Unsoundness in 'pub fn get'

### [Xudong-Huang/may](https://github.com/Xudong-Huang/may)

- clone: `https://github.com/Xudong-Huang/may.git`
- Issue 数：1；accepted-and-fixed：1
- [Xudong-Huang/may#112](https://github.com/Xudong-Huang/may/issues/112) Unsound Use of get_unchecked in wakeup Function · 已认可修复

### [apache/datafusion-comet](https://github.com/apache/datafusion-comet)

- clone: `https://github.com/apache/datafusion-comet.git`
- Issue 数：1；accepted-and-fixed：1
- [apache/datafusion-comet#1141](https://github.com/apache/datafusion-comet/issues/1141) Maybe Undefined Behavior in StringView::as_utf8_str · 已认可修复

### [arceos-org/arceos](https://github.com/arceos-org/arceos)

- clone: `https://github.com/arceos-org/arceos.git`
- Issue 数：1；accepted-and-fixed：1
- [arceos-org/arceos#204](https://github.com/arceos-org/arceos/issues/204) Unsoundness in run_cmd · 已认可修复

### [archseer/enigma](https://github.com/archseer/enigma)

- clone: `https://github.com/archseer/enigma.git`
- Issue 数：2；accepted-and-fixed：0
- [archseer/enigma#44](https://github.com/archseer/enigma/issues/44) Unsound problem in InstrPtr::module
- [archseer/enigma#45](https://github.com/archseer/enigma/issues/45) Unsoundness in cmp

### [asg017/sqlite-loadable-rs](https://github.com/asg017/sqlite-loadable-rs)

- clone: `https://github.com/asg017/sqlite-loadable-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [asg017/sqlite-loadable-rs#24](https://github.com/asg017/sqlite-loadable-rs/issues/24) Maybe unsound in set_argv_index and set_omit

### [astonbitecode/j4rs](https://github.com/astonbitecode/j4rs)

- clone: `https://github.com/astonbitecode/j4rs.git`
- Issue 数：2；accepted-and-fixed：2
- [astonbitecode/j4rs#138](https://github.com/astonbitecode/j4rs/issues/138) Unsoundness in to_rust_string may cause UB due to unsafe assumptions about the input pointer · 已认可修复
- [astonbitecode/j4rs#140](https://github.com/astonbitecode/j4rs/issues/140) Unsoundness in drop_c_string · 已认可修复

### [async-rs/async-std](https://github.com/async-rs/async-std)

- clone: `https://github.com/async-rs/async-std.git`
- Issue 数：1；accepted-and-fixed：0
- [async-rs/async-std#1103](https://github.com/async-rs/async-std/issues/1103) Maybe Multiple unsoundness problems

### [awslabs/snapchange](https://github.com/awslabs/snapchange)

- clone: `https://github.com/awslabs/snapchange.git`
- Issue 数：1；accepted-and-fixed：0
- [awslabs/snapchange#43](https://github.com/awslabs/snapchange/issues/43) Unsoundness in pub field "console_output"

### [aya-rs/aya](https://github.com/aya-rs/aya)

- clone: `https://github.com/aya-rs/aya.git`
- Issue 数：1；accepted-and-fixed：1
- [aya-rs/aya#1112](https://github.com/aya-rs/aya/issues/1112) Multiple unsoundness problems in aya-ebpf/programs · 已认可修复

### [bbodi/rustarok](https://github.com/bbodi/rustarok)

- clone: `https://github.com/bbodi/rustarok.git`
- Issue 数：1；accepted-and-fixed：0
- [bbodi/rustarok#25](https://github.com/bbodi/rustarok/issues/25) unsoundness in write_f32 groups

### [beling/bsuccinct-rs](https://github.com/beling/bsuccinct-rs)

- clone: `https://github.com/beling/bsuccinct-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [beling/bsuccinct-rs#11](https://github.com/beling/bsuccinct-rs/issues/11) possible unsound code

### [bennetthardwick/rust-obs-plugins](https://github.com/bennetthardwick/rust-obs-plugins)

- clone: `https://github.com/bennetthardwick/rust-obs-plugins.git`
- Issue 数：1；accepted-and-fixed：0
- [bennetthardwick/rust-obs-plugins#52](https://github.com/bennetthardwick/rust-obs-plugins/issues/52) Maybe unsound in VideoDataSourceContext::pointer

### [bergercookie/asm-lsp](https://github.com/bergercookie/asm-lsp)

- clone: `https://github.com/bergercookie/asm-lsp.git`
- Issue 数：1；accepted-and-fixed：1
- [bergercookie/asm-lsp#187](https://github.com/bergercookie/asm-lsp/issues/187) get_str and get_string Are Unsound Due to Unchecked UTF-8 Operations · 已认可修复

### [bluealloy/revm](https://github.com/bluealloy/revm)

- clone: `https://github.com/bluealloy/revm.git`
- Issue 数：1；accepted-and-fixed：1
- [bluealloy/revm#1902](https://github.com/bluealloy/revm/issues/1902) Maybe unsound in ExtBytecode · 已认可修复

### [bluss/arrayvec](https://github.com/bluss/arrayvec)

- clone: `https://github.com/bluss/arrayvec.git`
- Issue 数：1；accepted-and-fixed：0
- [bluss/arrayvec#300](https://github.com/bluss/arrayvec/issues/300) Maybe Multiple unsoundness problems

### [blyssprivacy/sdk](https://github.com/blyssprivacy/sdk)

- clone: `https://github.com/blyssprivacy/sdk.git`
- Issue 数：1；accepted-and-fixed：0
- [blyssprivacy/sdk#35](https://github.com/blyssprivacy/sdk/issues/35) Undefined Behavior in Db::get_nice_slice due to Unaligned Reference

### [bnb-chain/revm](https://github.com/bnb-chain/revm)

- clone: `https://github.com/bnb-chain/revm.git`
- Issue 数：1；accepted-and-fixed：0
- [bnb-chain/revm#89](https://github.com/bnb-chain/revm/issues/89) Maybe unsound in current_opcode

### [brson/rust-sdl](https://github.com/brson/rust-sdl)

- clone: `https://github.com/brson/rust-sdl.git`
- Issue 数：1；accepted-and-fixed：0
- [brson/rust-sdl#168](https://github.com/brson/rust-sdl/issues/168) Maybe unsound in Surface::raw

### [capnproto/capnproto-rust](https://github.com/capnproto/capnproto-rust)

- clone: `https://github.com/capnproto/capnproto-rust.git`
- Issue 数：1；accepted-and-fixed：1
- [capnproto/capnproto-rust#534](https://github.com/capnproto/capnproto-rust/issues/534) Maybe unsound in set_capability_pointer · 已认可修复

### [cea-sec/sanzu](https://github.com/cea-sec/sanzu)

- clone: `https://github.com/cea-sec/sanzu.git`
- Issue 数：1；accepted-and-fixed：0
- [cea-sec/sanzu#309](https://github.com/cea-sec/sanzu/issues/309) Unsoundness in AVFrame::plane

### [changlan/kytan](https://github.com/changlan/kytan)

- clone: `https://github.com/changlan/kytan.git`
- Issue 数：1；accepted-and-fixed：0
- [changlan/kytan#39](https://github.com/changlan/kytan/issues/39) Unsoundness in raw_cksum

### [chinedufn/swift-bridge](https://github.com/chinedufn/swift-bridge)

- clone: `https://github.com/chinedufn/swift-bridge.git`
- Issue 数：1；accepted-and-fixed：0
- [chinedufn/swift-bridge#304](https://github.com/chinedufn/swift-bridge/issues/304) Bug report: Unsoundness  in FfiSlice::as_slice

### [crabml/crabml](https://github.com/crabml/crabml)

- clone: `https://github.com/crabml/crabml.git`
- Issue 数：1；accepted-and-fixed：0
- [crabml/crabml#218](https://github.com/crabml/crabml/issues/218) Unsoundness issue in fn vec_fma_f16_f16_neon

### [danielpclark/rutie](https://github.com/danielpclark/rutie)

- clone: `https://github.com/danielpclark/rutie.git`
- Issue 数：1；accepted-and-fixed：0
- [danielpclark/rutie#190](https://github.com/danielpclark/rutie/issues/190) Unsoundness in parse_arguments

### [datenlord/async-rdma](https://github.com/datenlord/async-rdma)

- clone: `https://github.com/datenlord/async-rdma.git`
- Issue 数：1；accepted-and-fixed：0
- [datenlord/async-rdma#125](https://github.com/datenlord/async-rdma/issues/125) Maybe unsound issue in cm_connect_helper

### [dnglab/dnglab](https://github.com/dnglab/dnglab)

- clone: `https://github.com/dnglab/dnglab.git`
- Issue 数：1；accepted-and-fixed：0
- [dnglab/dnglab#488](https://github.com/dnglab/dnglab/issues/488) Not sure whether it is a UB?

### [dontpanic92/OpenPAL3](https://github.com/dontpanic92/OpenPAL3)

- clone: `https://github.com/dontpanic92/OpenPAL3.git`
- Issue 数：1；accepted-and-fixed：1
- [dontpanic92/OpenPAL3#51](https://github.com/dontpanic92/OpenPAL3/issues/51) Maybe unsound in add_ref · 已认可修复

### [douchuan/jvm](https://github.com/douchuan/jvm)

- clone: `https://github.com/douchuan/jvm.git`
- Issue 数：2；accepted-and-fixed：0
- [douchuan/jvm#13](https://github.com/douchuan/jvm/issues/13) Unsound Use of std::str::from_utf8_unchecked in find_symbol
- [douchuan/jvm#14](https://github.com/douchuan/jvm/issues/14) Unsoundness in load_class_from_path

### [dragonflyoss/nydus](https://github.com/dragonflyoss/nydus)

- clone: `https://github.com/dragonflyoss/nydus.git`
- Issue 数：1；accepted-and-fixed：0
- [dragonflyoss/nydus#1814](https://github.com/dragonflyoss/nydus/issues/1814) [Unsoundness] Undefined Behavior in public API alloc_buf due to uninitialized memory usage

### [eclipse-zenoh/zenoh](https://github.com/eclipse-zenoh/zenoh)

- clone: `https://github.com/eclipse-zenoh/zenoh.git`
- Issue 数：2；accepted-and-fixed：0
- [eclipse-zenoh/zenoh#1643](https://github.com/eclipse-zenoh/zenoh/issues/1643) Unsoundness in OwnedDescriptor::atomic
- [eclipse-zenoh/zenoh#1683](https://github.com/eclipse-zenoh/zenoh/issues/1683) Maybe unsound in watchdog::descriptor::confirm

### [everx-labs/ever-sdk](https://github.com/everx-labs/ever-sdk)

- clone: `https://github.com/everx-labs/ever-sdk.git`
- Issue 数：1；accepted-and-fixed：0
- [everx-labs/ever-sdk#1062](https://github.com/everx-labs/ever-sdk/issues/1062) Maybe unsound in StringData

### [evmar/n2](https://github.com/evmar/n2)

- clone: `https://github.com/evmar/n2.git`
- Issue 数：3；accepted-and-fixed：0
- [evmar/n2#121](https://github.com/evmar/n2/issues/121) Bug Report: Invalid Input Handling in `Scanner::slice` Method
- [evmar/n2#122](https://github.com/evmar/n2/issues/122) Bug Report: Out of Bounds Access in `Scanner::peek` Method
- [evmar/n2#123](https://github.com/evmar/n2/issues/123) Bug Report: Invalid Input Handling in `Scanner::peek_newline` Method

### [evmar/retrowin32](https://github.com/evmar/retrowin32)

- clone: `https://github.com/evmar/retrowin32.git`
- Issue 数：2；accepted-and-fixed：0
- [evmar/retrowin32#72](https://github.com/evmar/retrowin32/issues/72) Unsoundness in safe 'copy' function may cause UB due to lack of bounds checks
- [evmar/retrowin32#74](https://github.com/evmar/retrowin32/issues/74) Arg<T> Implementation Allows Undefined Behavior Due to Unsafe Pointer Operations

### [flaneur2020/pua-lang](https://github.com/flaneur2020/pua-lang)

- clone: `https://github.com/flaneur2020/pua-lang.git`
- Issue 数：1；accepted-and-fixed：1
- [flaneur2020/pua-lang#76](https://github.com/flaneur2020/pua-lang/issues/76) dealloc Function Causes Undefined Behavior Due to Improper Use of Vec::from_raw_parts · 已认可修复

### [fugue-re/fugue-core](https://github.com/fugue-re/fugue-core)

- clone: `https://github.com/fugue-re/fugue-core.git`
- Issue 数：1；accepted-and-fixed：1
- [fugue-re/fugue-core#18](https://github.com/fugue-re/fugue-core/issues/18) Unsound Use of get_unchecked in unchecked_symbol · 已认可修复

### [georust/gdal](https://github.com/georust/gdal)

- clone: `https://github.com/georust/gdal.git`
- Issue 数：1；accepted-and-fixed：1
- [georust/gdal#588](https://github.com/georust/gdal/issues/588) Unsoundness in "_string" and "_pathbuf" · 已认可修复

### [gluon-lang/gluon](https://github.com/gluon-lang/gluon)

- clone: `https://github.com/gluon-lang/gluon.git`
- Issue 数：1；accepted-and-fixed：0
- [gluon-lang/gluon#969](https://github.com/gluon-lang/gluon/issues/969) WriteOnly::write can cause stack overflow or out-of-memory error with large inputs

### [hermit-os/uhyve](https://github.com/hermit-os/uhyve)

- clone: `https://github.com/hermit-os/uhyve.git`
- Issue 数：2；accepted-and-fixed：1
- [hermit-os/uhyve#834](https://github.com/hermit-os/uhyve/issues/834) Maybe unsound in Vring::_flags · 已认可修复
- [hermit-os/uhyve#855](https://github.com/hermit-os/uhyve/issues/855) Maybe unsound in virtio's write_ fn groups

### [imazen/imageflow](https://github.com/imazen/imageflow)

- clone: `https://github.com/imazen/imageflow.git`
- Issue 数：1；accepted-and-fixed：0
- [imazen/imageflow#655](https://github.com/imazen/imageflow/issues/655) Maybe unsound in ffi::BitmapBgra pixels field

### [inherd/coco](https://github.com/inherd/coco)

- clone: `https://github.com/inherd/coco.git`
- Issue 数：1；accepted-and-fixed：0
- [inherd/coco#101](https://github.com/inherd/coco/issues/101) Unsoundness in run_ctags

### [jeaye/ncurses-rs](https://github.com/jeaye/ncurses-rs)

- clone: `https://github.com/jeaye/ncurses-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [jeaye/ncurses-rs#224](https://github.com/jeaye/ncurses-rs/issues/224) Unsoundness in ptr_to_string may cause UB due to unsafe assumptions about pointer validity and string encoding

### [jfecher/ante](https://github.com/jfecher/ante)

- clone: `https://github.com/jfecher/ante.git`
- Issue 数：1；accepted-and-fixed：1
- [jfecher/ante#221](https://github.com/jfecher/ante/issues/221) [Unsoundness] Critical soundness violations in public API src/util/trustme.rs (make_mut_ref, extend_lifetime) · 已认可修复

### [joaoviictorti/shadow](https://github.com/joaoviictorti/shadow)

- clone: `https://github.com/joaoviictorti/shadow.git`
- Issue 数：1；accepted-and-fixed：1
- [joaoviictorti/shadow#5](https://github.com/joaoviictorti/shadow/issues/5) Maybe unsound in add_remove_port_toggle · 已认可修复

### [kahing/catfs](https://github.com/kahing/catfs)

- clone: `https://github.com/kahing/catfs.git`
- Issue 数：1；accepted-and-fixed：0
- [kahing/catfs#77](https://github.com/kahing/catfs/issues/77) [Unsoundness] Critical: Public API make_self allows arbitrary lifetime extension leading to Use-After-Free

### [kcl-lang/kcl](https://github.com/kcl-lang/kcl)

- clone: `https://github.com/kcl-lang/kcl.git`
- Issue 数：1；accepted-and-fixed：1
- [kcl-lang/kcl#1776](https://github.com/kcl-lang/kcl/issues/1776) Maybe unsound in c2str_vec · 已认可修复

### [laptou/jsi-rs](https://github.com/laptou/jsi-rs)

- clone: `https://github.com/laptou/jsi-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [laptou/jsi-rs#5](https://github.com/laptou/jsi-rs/issues/5) Maybe unsound in new_unchecked

### [leetvr/hotham](https://github.com/leetvr/hotham)

- clone: `https://github.com/leetvr/hotham.git`
- Issue 数：1；accepted-and-fixed：0
- [leetvr/hotham#449](https://github.com/leetvr/hotham/issues/449) [Bug]: Unsoundness in u8_to_u32

### [leostera/lam](https://github.com/leostera/lam)

- clone: `https://github.com/leostera/lam.git`
- Issue 数：1；accepted-and-fixed：0
- [leostera/lam#9](https://github.com/leostera/lam/issues/9) Unsound issue in program

### [m4b/dryad](https://github.com/m4b/dryad)

- clone: `https://github.com/m4b/dryad.git`
- Issue 数：1；accepted-and-fixed：0
- [m4b/dryad#6](https://github.com/m4b/dryad/issues/6) Potential Undefined Behavior in from_executable Due to Unsafe Pointer Usage and Assumptions

### [maekawatoshiki/ferrugo](https://github.com/maekawatoshiki/ferrugo)

- clone: `https://github.com/maekawatoshiki/ferrugo.git`
- Issue 数：1；accepted-and-fixed：0
- [maekawatoshiki/ferrugo#2](https://github.com/maekawatoshiki/ferrugo/issues/2) Maybe unsound in CodeAttribute

### [maekawatoshiki/rapidus](https://github.com/maekawatoshiki/rapidus)

- clone: `https://github.com/maekawatoshiki/rapidus.git`
- Issue 数：1；accepted-and-fixed：0
- [maekawatoshiki/rapidus#72](https://github.com/maekawatoshiki/rapidus/issues/72) Maybe unsound in cstrp_to_str

### [meh/rust-ffmpeg](https://github.com/meh/rust-ffmpeg)

- clone: `https://github.com/meh/rust-ffmpeg.git`
- Issue 数：1；accepted-and-fixed：0
- [meh/rust-ffmpeg#188](https://github.com/meh/rust-ffmpeg/issues/188) Maybe unsound in RateIter::next

### [michelhe/rustboyadvance-ng](https://github.com/michelhe/rustboyadvance-ng)

- clone: `https://github.com/michelhe/rustboyadvance-ng.git`
- Issue 数：1；accepted-and-fixed：0
- [michelhe/rustboyadvance-ng#177](https://github.com/michelhe/rustboyadvance-ng/issues/177) Maybe unsound in WeakPointer::new

### [microsoft/aici](https://github.com/microsoft/aici)

- clone: `https://github.com/microsoft/aici.git`
- Issue 数：1；accepted-and-fixed：0
- [microsoft/aici#119](https://github.com/microsoft/aici/issues/119) maybe unsound in busy_reset

### [moratorium08/osmium](https://github.com/moratorium08/osmium)

- clone: `https://github.com/moratorium08/osmium.git`
- Issue 数：2；accepted-and-fixed：0
- [moratorium08/osmium#3](https://github.com/moratorium08/osmium/issues/3) Maybe unsound in new?
- [moratorium08/osmium#4](https://github.com/moratorium08/osmium/issues/4) Unsoundness in receive_data

### [mstange/samply](https://github.com/mstange/samply)

- clone: `https://github.com/mstange/samply.git`
- Issue 数：1；accepted-and-fixed：0
- [mstange/samply#719](https://github.com/mstange/samply/issues/719) [Unsoundness] Out-of-bounds read in public API parse_unk_size_null_utf16_string due to unsafe loop condition order

### [mycelial/mycelite](https://github.com/mycelial/mycelite)

- clone: `https://github.com/mycelial/mycelite.git`
- Issue 数：1；accepted-and-fixed：0
- [mycelial/mycelite#52](https://github.com/mycelial/mycelite/issues/52) Unsoundness in PtrIter::next

### [nash-io/openlimits](https://github.com/nash-io/openlimits)

- clone: `https://github.com/nash-io/openlimits.git`
- Issue 数：1；accepted-and-fixed：0
- [nash-io/openlimits#204](https://github.com/nash-io/openlimits/issues/204) Unsound Use of CStr::from_ptr in new Function

### [nervosnetwork/ckb-vm](https://github.com/nervosnetwork/ckb-vm)

- clone: `https://github.com/nervosnetwork/ckb-vm.git`
- Issue 数：1；accepted-and-fixed：1
- [nervosnetwork/ckb-vm#444](https://github.com/nervosnetwork/ckb-vm/issues/444) Unsoundness in cast_ptr_to_slice · 已认可修复

### [ninja-quant/ninjabook](https://github.com/ninja-quant/ninjabook)

- clone: `https://github.com/ninja-quant/ninjabook.git`
- Issue 数：1；accepted-and-fixed：0
- [ninja-quant/ninjabook#3](https://github.com/ninja-quant/ninjabook/issues/3) Unsound Implementation of get_mut in Orderbook Allows Undefined Behavior

### [nixops4/nixops4](https://github.com/nixops4/nixops4)

- clone: `https://github.com/nixops4/nixops4.git`
- Issue 数：1；accepted-and-fixed：1
- [nixops4/nixops4#65](https://github.com/nixops4/nixops4/issues/65) Maybe unsound in new_raw_clone · 已认可修复

### [nuta/kerla](https://github.com/nuta/kerla)

- clone: `https://github.com/nuta/kerla.git`
- Issue 数：1；accepted-and-fixed：0
- [nuta/kerla#170](https://github.com/nuta/kerla/issues/170) Unsoundness in fn parse_str_field

### [o8vm/krabs](https://github.com/o8vm/krabs)

- clone: `https://github.com/o8vm/krabs.git`
- Issue 数：1；accepted-and-fixed：0
- [o8vm/krabs#30](https://github.com/o8vm/krabs/issues/30) Unsoundness in

### [occlum/occlum](https://github.com/occlum/occlum)

- clone: `https://github.com/occlum/occlum.git`
- Issue 数：1；accepted-and-fixed：0
- [occlum/occlum#1645](https://github.com/occlum/occlum/issues/1645) [BUG] Maybe unsound in from_raw_ptr

### [only-cliches/NoProto](https://github.com/only-cliches/NoProto)

- clone: `https://github.com/only-cliches/NoProto.git`
- Issue 数：1；accepted-and-fixed：0
- [only-cliches/NoProto#25](https://github.com/only-cliches/NoProto/issues/25) Multiple Unsoundness issue

### [oreboot/oreboot](https://github.com/oreboot/oreboot)

- clone: `https://github.com/oreboot/oreboot.git`
- Issue 数：1；accepted-and-fixed：0
- [oreboot/oreboot#776](https://github.com/oreboot/oreboot/issues/776) Unsoundness issue in

### [perpetual-ml/perpetual](https://github.com/perpetual-ml/perpetual)

- clone: `https://github.com/perpetual-ml/perpetual.git`
- Issue 数：1；accepted-and-fixed：0
- [perpetual-ml/perpetual#34](https://github.com/perpetual-ml/perpetual/issues/34) Unsound Use of get_unchecked in from_parent_child and from_parent_two_children

### [pipedown/noise](https://github.com/pipedown/noise)

- clone: `https://github.com/pipedown/noise.git`
- Issue 数：1；accepted-and-fixed：0
- [pipedown/noise#82](https://github.com/pipedown/noise/issues/82) Unsoundness in maybe_add_value

### [pleco-rs/Pleco](https://github.com/pleco-rs/Pleco)

- clone: `https://github.com/pleco-rs/Pleco.git`
- Issue 数：1；accepted-and-fixed：1
- [pleco-rs/Pleco#163](https://github.com/pleco-rs/Pleco/issues/163) Unsoundness in pleco_engine/src/root_moves/root_moves_list.rs "insert_score_depth" and "insert_score" · 已认可修复

### [posit-dev/ark](https://github.com/posit-dev/ark)

- clone: `https://github.com/posit-dev/ark.git`
- Issue 数：2；accepted-and-fixed：0
- [posit-dev/ark#653](https://github.com/posit-dev/ark/issues/653) Maybe unsound in RLocal::new
- [posit-dev/ark#729](https://github.com/posit-dev/ark/issues/729) Soundness Issue: Unsafe raw pointer operation exposed through safe public API r_show_message

### [r9os/r9](https://github.com/r9os/r9)

- clone: `https://github.com/r9os/r9.git`
- Issue 数：1；accepted-and-fixed：0
- [r9os/r9#51](https://github.com/r9os/r9/issues/51) Maybe unsound in write_reg groups

### [raskr/rust-autograd](https://github.com/raskr/rust-autograd)

- clone: `https://github.com/raskr/rust-autograd.git`
- Issue 数：1；accepted-and-fixed：0
- [raskr/rust-autograd#65](https://github.com/raskr/rust-autograd/issues/65) Unsoundness in gen_random_array

### [rcore-os/zCore](https://github.com/rcore-os/zCore)

- clone: `https://github.com/rcore-os/zCore.git`
- Issue 数：1；accepted-and-fixed：0
- [rcore-os/zCore#406](https://github.com/rcore-os/zCore/issues/406) unsoundness in nvme_configure_admin_queue(&mut self)

### [rust-bio/rust-htslib](https://github.com/rust-bio/rust-htslib)

- clone: `https://github.com/rust-bio/rust-htslib.git`
- Issue 数：1；accepted-and-fixed：0
- [rust-bio/rust-htslib#459](https://github.com/rust-bio/rust-htslib/issues/459) Multiple unsound problems in bcf crate

### [rustcoreutils/posixutils-rs](https://github.com/rustcoreutils/posixutils-rs)

- clone: `https://github.com/rustcoreutils/posixutils-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [rustcoreutils/posixutils-rs#380](https://github.com/rustcoreutils/posixutils-rs/issues/380) Unsoundness in open_at function

### [sciter-sdk/rust-sciter](https://github.com/sciter-sdk/rust-sciter)

- clone: `https://github.com/sciter-sdk/rust-sciter.git`
- Issue 数：1；accepted-and-fixed：0
- [sciter-sdk/rust-sciter#150](https://github.com/sciter-sdk/rust-sciter/issues/150) Unsoundness in get

### [servo/mozjs](https://github.com/servo/mozjs)

- clone: `https://github.com/servo/mozjs.git`
- Issue 数：1；accepted-and-fixed：0
- [servo/mozjs#531](https://github.com/servo/mozjs/issues/531) Maybe unsound in CustomAutoRooterGuard::new

### [sgmarz/osblog](https://github.com/sgmarz/osblog)

- clone: `https://github.com/sgmarz/osblog.git`
- Issue 数：1；accepted-and-fixed：0
- [sgmarz/osblog#40](https://github.com/sgmarz/osblog/issues/40) Unsoundness in do_syscall

### [signalapp/ringrtc](https://github.com/signalapp/ringrtc)

- clone: `https://github.com/signalapp/ringrtc.git`
- Issue 数：1；accepted-and-fixed：0
- [signalapp/ringrtc#60](https://github.com/signalapp/ringrtc/issues/60) Possible unsoundness in rtc_Bytes::as_slice

### [smol-rs/async-task](https://github.com/smol-rs/async-task)

- clone: `https://github.com/smol-rs/async-task.git`
- Issue 数：1；accepted-and-fixed：0
- [smol-rs/async-task#89](https://github.com/smol-rs/async-task/issues/89) Maybe Multiple unsoundness problems

### [starcoinorg/starcoin](https://github.com/starcoinorg/starcoin)

- clone: `https://github.com/starcoinorg/starcoin.git`
- Issue 数：1；accepted-and-fixed：0
- [starcoinorg/starcoin#4763](https://github.com/starcoinorg/starcoin/issues/4763) [Unsoundness] Potential buffer over-read in public API cryptonight_r due to unchecked size parameter

### [tensorbase/tensorbase](https://github.com/tensorbase/tensorbase)

- clone: `https://github.com/tensorbase/tensorbase.git`
- Issue 数：2；accepted-and-fixed：0
- [tensorbase/tensorbase#351](https://github.com/tensorbase/tensorbase/issues/351) Unsound Use of CString::from_vec_unchecked in bytes_to_cstring
- [tensorbase/tensorbase#352](https://github.com/tensorbase/tensorbase/issues/352) Public API “shape_slice” “shape_vec” “shape_vec_u8” constructs unaligned references causing Undefined Behavior

### [tensorchord/pgvecto.rs](https://github.com/tensorchord/pgvecto.rs)

- clone: `https://github.com/tensorchord/pgvecto.rs.git`
- Issue 数：1；accepted-and-fixed：0
- [tensorchord/pgvecto.rs#653](https://github.com/tensorchord/pgvecto.rs/issues/653) Public API try_pod_read_unaligned causes Out-of-Bounds read due to missing length check

### [topiary/topiary](https://github.com/topiary/topiary)

- clone: `https://github.com/topiary/topiary.git`
- Issue 数：1；accepted-and-fixed：0
- [topiary/topiary#792](https://github.com/topiary/topiary/issues/792) Unsoundness in 'parse','child_by_field_name' and 'field_id_for_name'

### [toxoidengine/toxoid](https://github.com/toxoidengine/toxoid)

- clone: `https://github.com/toxoidengine/toxoid.git`
- Issue 数：1；accepted-and-fixed：0
- [toxoidengine/toxoid#5](https://github.com/toxoidengine/toxoid/issues/5) Maybe Unsound in init_image

### [tursodatabase/libsql](https://github.com/tursodatabase/libsql)

- clone: `https://github.com/tursodatabase/libsql.git`
- Issue 数：2；accepted-and-fixed：1
- [tursodatabase/libsql#1859](https://github.com/tursodatabase/libsql/issues/1859) Unsound Use of str::from_utf8_unchecked in keyword_token · 已认可修复
- [tursodatabase/libsql#1873](https://github.com/tursodatabase/libsql/issues/1873) Maybe unsound in db_version::next_db_version

### [tylerwhall/zephyr-rust](https://github.com/tylerwhall/zephyr-rust)

- clone: `https://github.com/tylerwhall/zephyr-rust.git`
- Issue 数：1；accepted-and-fixed：0
- [tylerwhall/zephyr-rust#58](https://github.com/tylerwhall/zephyr-rust/issues/58) Maybe unsound in free_mutex

### [vinc/moros](https://github.com/vinc/moros)

- clone: `https://github.com/vinc/moros.git`
- Issue 数：1；accepted-and-fixed：0
- [vinc/moros#724](https://github.com/vinc/moros/issues/724) Unsound issue in sys::process::exec

### [workflow-rs/workflow-rs](https://github.com/workflow-rs/workflow-rs)

- clone: `https://github.com/workflow-rs/workflow-rs.git`
- Issue 数：1；accepted-and-fixed：0
- [workflow-rs/workflow-rs#14](https://github.com/workflow-rs/workflow-rs/issues/14) unsoundness in buffer_as_slice and buffer_as_slice_mut

### [xajik/rust-cross-platform-mobile](https://github.com/xajik/rust-cross-platform-mobile)

- clone: `https://github.com/xajik/rust-cross-platform-mobile.git`
- Issue 数：1；accepted-and-fixed：0
- [xajik/rust-cross-platform-mobile#2](https://github.com/xajik/rust-cross-platform-mobile/issues/2) Maybe unsound in JavaString::new

### [yamadapc/augmented-audio](https://github.com/yamadapc/augmented-audio)

- clone: `https://github.com/yamadapc/augmented-audio.git`
- Issue 数：1；accepted-and-fixed：0
- [yamadapc/augmented-audio#615](https://github.com/yamadapc/augmented-audio/issues/615) Maybe unsound in SharedGeneric::drop

### [zackradisic/rust-vs-zig](https://github.com/zackradisic/rust-vs-zig)

- clone: `https://github.com/zackradisic/rust-vs-zig.git`
- Issue 数：1；accepted-and-fixed：0
- [zackradisic/rust-vs-zig#2](https://github.com/zackradisic/rust-vs-zig/issues/2) Maybe unsound in ObjClosure

### [zhihu/rucene](https://github.com/zhihu/rucene)

- clone: `https://github.com/zhihu/rucene.git`
- Issue 数：3；accepted-and-fixed：0
- [zhihu/rucene#19](https://github.com/zhihu/rucene/issues/19) Unsoundness in BytesRefHash::pool
- [zhihu/rucene#20](https://github.com/zhihu/rucene/issues/20) Unsoundness in UtilMSBIntroSorter::reader
- [zhihu/rucene#21](https://github.com/zhihu/rucene/issues/21) Unsoundness in read_data

### [zmwangx/rust-ffmpeg](https://github.com/zmwangx/rust-ffmpeg)

- clone: `https://github.com/zmwangx/rust-ffmpeg.git`
- Issue 数：1；accepted-and-fixed：0
- [zmwangx/rust-ffmpeg#225](https://github.com/zmwangx/rust-ffmpeg/issues/225) Multiple Unsoundness problems in codec

### [zzau13/yarte](https://github.com/zzau13/yarte)

- clone: `https://github.com/zzau13/yarte.git`
- Issue 数：1；accepted-and-fixed：0
- [zzau13/yarte#319](https://github.com/zzau13/yarte/issues/319) Unsoundness in get_nodes

## advisory-db 有效 Issue（保留账本，不作为扫描仓）

- [rustsec/advisory-db#2140](https://github.com/rustsec/advisory-db/issues/2140) Corundum is unmaintained and may unsoundness
- [rustsec/advisory-db#2141](https://github.com/rustsec/advisory-db/issues/2141) Unsoundly mark unsafe function as safe in crate Ngrammatic
- [rustsec/advisory-db#2146](https://github.com/rustsec/advisory-db/issues/2146) Unsoundness in Pleco crates
- [rustsec/advisory-db#2149](https://github.com/rustsec/advisory-db/issues/2149) Unsoundness in perpetual
- [rustsec/advisory-db#2150](https://github.com/rustsec/advisory-db/issues/2150) Unsoundness in ninjabook
- [rustsec/advisory-db#2151](https://github.com/rustsec/advisory-db/issues/2151) Unsoundness in swift-bridge

## 不进入扫描的 Issue

### 范围排除
- [babalae/better-genshin-impact#2192](https://github.com/babalae/better-genshin-impact/issues/2192) [question] 月落银地图追踪的脚本在哪下载 — 窗口内创建，但不是 unsoundness 上报
- [endorlabs/MIRAI#12](https://github.com/endorlabs/MIRAI/issues/12) How to generate Rust program's ICFG — 窗口内创建，但不是 unsoundness 上报

### 明确反驳或作者撤回
- [Amanieu/corosensei#45](https://github.com/Amanieu/corosensei/issues/45) Maybe unsound in ValgrindStackRegistration::new — 维护者解释该调用在项目语义下不构成 UB，作者随后同意。
- [BurntSushi/byteorder#219](https://github.com/BurntSushi/byteorder/issues/219) unsoundness in src/lib.rs — 维护者解释输入切片已经有效，作者明确承认报告错误。
- [BurntSushi/byteorder#220](https://github.com/BurntSushi/byteorder/issues/220) unsoundness in write_i16_into — 维护者指出这是 #219 的完全重复报告；沿用同一明确反驳。
- [h33p/cglue#20](https://github.com/h33p/cglue/issues/20) Undefined Behavior in CVec::insert Leading to Memory Corruption — 作者明确撤回：崩溃来自测试中的巨额分配，与 CVec::insert 无关。
- [h33p/cglue#21](https://github.com/h33p/cglue/issues/21) Undefined Behavior in CVec::insert Leading to Memory Corruption — 与 #20 的正文和 PoC 相同；沿用作者对同一报告的明确撤回。
- [intel/tsffs#123](https://github.com/intel/tsffs/issues/123) Unsound Use of CStr::from_ptr in add_architecture_hint — 作者复查后明确撤回 UB 主张，改称只需要 safety 文档。
- [oreboot/oreboot#761](https://github.com/oreboot/oreboot/issues/761) maybe unsound in print_mem and print_strmem — 维护者明确表示论证不适用于该固件代码并以 not applicable 关闭。
- [servo/ipc-channel#375](https://github.com/servo/ipc-channel/issues/375) Bug report: unsoundness in OsIpcSharedMemory::from_bytes — 作者明确撤回：崩溃来自巨额分配，不是 unsoundness。
- [wgpu-mc/wgpu-mc#114](https://github.com/wgpu-mc/wgpu-mc/issues/114) Maybe unsound in setIndexBuffer — 维护者指出 i32 数组的对齐不是报告所称的 1 字节，否定该 unsoundness 论证。
- [zakarumych/allocator-api2#43](https://github.com/zakarumych/allocator-api2/issues/43) Maybe Multiple unsoundness problems — 维护者解释 ptr::write 的前置条件已满足，作者明确确认代码 sound。

