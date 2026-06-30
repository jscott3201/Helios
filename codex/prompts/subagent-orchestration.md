# Subagent Orchestration Prompt

```text
Spawn focused subagents for this milestone and wait for all results before making changes:

1. codebase-mapper: map existing files/symbols relevant to <scope>; read-only.
2. external-researcher: verify current APIs/docs for <MLX/Gemma/Codex/etc.>; cite exact sources.
3. performance-analyst: identify benchmark/profiling surfaces and command plan; no source edits.
4. test-verifier: identify tests needed for acceptance; run safe tests if available.

Each subagent must return evidence, gaps, and recommended next action. Parent agent must synthesize conflicts and decide the implementation path.
```


Use `tui_ux_engineer` for Ratatui operator console design/review, especially terminal lifecycle, page/provider architecture, and snapshot/usability tests.
