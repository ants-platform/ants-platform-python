---
name: ralph
description: >
  Run the fresh-context Ralph loop over a Spec Kit feature's tasks.md - a new
  `claude -p` process per task (implement one task, run gates, mark [x], commit,
  exit, repeat). OPT-IN, NEW/GREENFIELD FEATURES ONLY. Spec Kit
  (specify->plan->tasks->analyze) is the standard for everything; this only wraps
  the implement phase. Use when a developer asks to "ralph", "run the ralph loop",
  "loop over tasks", or autonomously implement a planned greenfield feature.
---

# Ralph loop (opt-in, greenfield only)

This skill drives `.specify/ralph/ralph.sh`. Do NOT reimplement the loop in-session
(that defeats fresh context). Your job is to validate fit, set it up safely, and
launch the harness.

## When to REFUSE (and say why)

- Not a new/greenfield feature (e.g. ad-hoc edits to the existing TS monorepo).
- No `tasks.md` / hasn't been through `/speckit-plan` + `/speckit-tasks`.
- Fewer than ~10 tasks, or no real test gates -> tell them plain `/speckit-implement`
  or Plan Mode is the better tool.
- Target repo's pre-commit hook does a whole-repo check that will fail on
  pre-existing debt (e.g. lf-fork Prettier on TS) - only safe when the feature's
  files are outside that glob (a new Go/Python service is fine).

## Steps

1. Resolve the feature dir (`.specify/feature.json` or ask). Confirm `plan.md` +
   `tasks.md` exist and there are unchecked tasks.
2. Ensure a dedicated branch `ralph/<feature>` exists and is checked out
   (`git switch -c ralph/<feature>`). NEVER run on develop/main.
3. Determine the quality gates for the feature (e.g. `go vet ./... && go test ./...`).
4. Run a **dry run** first and show the user the resolved config:
   `.specify/ralph/ralph.sh --feature <dir> --gates "<cmd>" --dry-run`
5. On approval, launch the real loop (drop `--dry-run`). It is bounded by
   `--max-iterations` (default unchecked+10), stuck-detection, and the completion
   token. It commits per task and NEVER pushes.
6. When it stops, summarise: tasks completed, iterations used, anything that
   needs a human, and remind them to review the branch + open a PR.

## Guardrails to preserve

- One task per iteration; gates are the proof of done, not the checkbox.
- Never push; never touch develop/main; never run destructive DB/infra commands.
- Mandatory human PR review after the run.
