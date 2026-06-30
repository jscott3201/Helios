# Research and Review Agent Skills v1.0.0

This package contains reusable, generalized agent skills and subagent definitions for research and review workflows across any application or codebase. It is intentionally not tied to a specific repository, programming language, framework, or product domain.

The package is split into platform-native install folders:

- `codex/` — Codex-compatible skills in `.agents/skills/` and custom agents in `.codex/agents/`.
- `cursor/` — Cursor-compatible skills in `.cursor/skills/` and subagents in `.cursor/agents/`.
- `shared/` — portable templates, rubrics, and playbooks that can be copied into either environment or shared with colleagues.

## Skill set

| Skill | Purpose |
|---|---|
| `codebase-research-pass` | Structured codebase research before planning, review, or implementation. |
| `external-source-research` | Research against authoritative external docs, standards, APIs, SDKs, and release notes. |
| `spec-contract-compliance-review` | Trace requirements from specs/contracts/policies to code and tests. |
| `architecture-design-review` | Review architecture, refactors, dependency boundaries, and maintainability tradeoffs. |
| `multi-agent-pr-review` | Review PRs/patches/diffs through correctness, security, test, docs/API, and release lenses. |
| `performance-ab-benchmark-review` | Establish reproducible performance baselines and A/B benchmark evidence. |
| `release-readiness-review` | Decide whether a change is ready to merge, ship, publish, or hand off. |

## Subagent set

| Subagent | Purpose |
|---|---|
| `codebase-mapper` | Read-only architecture and execution-path exploration. |
| `external-researcher` | Primary-source documentation, spec, API, and dependency research. |
| `spec-tracer` | Requirement extraction and traceability matrix creation. |
| `correctness-reviewer` | Functional review for bugs, regressions, compatibility, and missing tests. |
| `security-reliability-reviewer` | Security, privacy, concurrency, failure-mode, and fail-closed review. |
| `performance-analyst` | Benchmark inventory, profiling, A/B comparison, and interpretation. |
| `test-verifier` | Independent build/test/lint/typecheck/CI verification. |
| `release-risk-reviewer` | Merge/release readiness, rollout, migration, docs, observability, and rollback risk. |

## Recommended installation

For Codex, copy the contents of `codex/` into the repository root or your user-level configuration. For Cursor, copy the contents of `cursor/` into the repository root or your user-level configuration. Do not install both `codex/` and `cursor/` folders into the same repo unless you intentionally want both platform-specific copies present.

## Design goals

- Generalizable to any codebase.
- Evidence-led and skeptical by default.
- Read-only until explicitly asked to change code.
- Subagent-driven for long research, parallel review, and independent verification.
- Small, focused skills with explicit trigger descriptions and output contracts.

See `docs/usage-playbook.md` for recommended prompts and `docs/platform-format-notes.md` for platform-specific format notes.
