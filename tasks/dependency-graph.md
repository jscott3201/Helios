# Dependency Graph

```mermaid
graph TD
  M00[M00 repo bootstrap] --> M01[M01 native MLX loader]
  M00 --> M02[M02 tokenizer/chat/config]
  M01 --> M03[M03 greedy inference]
  M02 --> M03
  M03 --> M04[M04 parity harness]
  M04 --> M05[M05 TUI operator console]
  M04 --> M06[M06 MTP]
  M05 --> M06
  M03 --> M07[M07 KV core]
  M05 --> M07
  M07 --> M08[M08 SSD cache]
  M08 --> M09[M09 KV compression]
  M02 --> M10[M10 adapters]
  M03 --> M10
  M07 --> M10
  M05 --> M10
  M03 --> M11[M11 server]
  M05 --> M11
  M06 --> M12[M12 tiny16 release]
  M08 --> M12
  M10 --> M12
  M11 --> M12
```

If Mermaid is not rendered in your environment, treat this as the dependency order:

```text
M00 -> {M01, M02} -> M03 -> M04 -> M05
M04 + M05 -> M06
M03 + M05 -> M07 -> M08 -> M09
{M02, M03, M05, M07} -> M10
{M03, M05} -> M11
{M06, M08, M10, M11} -> M12
```
