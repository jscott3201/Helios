# Backlog

Use this backlog only after reading the milestone files. Do not treat this as permission to implement everything at once.

## Foundation

- M00-T01 workspace skeleton
- M00-T02 Rust 1.95.0 pin
- M00-T03 native CMake skeleton
- M00-T04 verification command

## Native runtime

- M01-T01 FFI status/error model
- M01-T02 target opaque handle lifecycle
- M01-T03 Rust safe wrappers
- M03-T01 target model load
- M03-T02 prefill/decode-one

## Correctness

- M02-T01 Gemma config validator
- M02-T02 tokenizer fixtures
- M02-T03 chat compiler
- M04-T01 reference harness
- M04-T02 token diff tooling

## TUI/operator UX

- M05-T01 Ratatui crate skeleton
- M05-T02 event loop and terminal lifecycle
- M05-T03 provider trait with mock/file providers
- M05-T04 dashboard/config/benchmark/log/help pages
- M05-T05 snapshot, reducer, keybinding tests
- M05-T06 TUI usability report

## MTP

- M06-T01 drafter load
- M06-T02 draft/verify/accept loop
- M06-T03 rollback exactness tests
- M06-T04 acceptance metrics
- M06-T05 TUI MTP page integration hook

## KV/cache

- M07-T01 logical block metadata
- M07-T02 RAM prefix cache
- M08-T01 SSD manifest
- M08-T02 checksum restore
- M09-T01 q8/q4 prefix compression
- M09-T02 Planar/Iso experiment gate
- M09-T03 TUI cache page integration hook

## Adapters

- M10-T01 PEFT manifest parser
- M10-T02 safetensors adapter loader
- M10-T03 one active adapter per request
- M10-T04 adapter-aware cache namespace tests
- M10-T05 TUI adapter page integration hook

## Server/release

- M11-T01 local server
- M11-T02 streaming chat
- M11-T03 metrics
- M11-T04 TUI HttpProvider/live attach
- M12-T01 tiny16 benchmark campaign
- M12-T02 TUI-driven release walkthrough
- M12-T03 release risk review
