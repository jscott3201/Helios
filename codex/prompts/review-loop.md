# Review / Optimization Loop Prompt

```text
Use $multi-agent-pr-review and $performance-ab-benchmark-review to review the current branch against the active milestone.
Spawn correctness, security/reliability, performance, and release-risk reviewers as separate subagents.
Score findings as blocker/high/medium/low/info.
Do not accept performance wins without correctness and variance evidence.
End with a merge/readiness decision and the smallest set of changes required before proceeding.
```
