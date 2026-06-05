# Ralph Loop - playbook

**Opt-in. New / greenfield features only.** Spec Kit is the standard for every
change; Ralph is a power tool a developer reaches for when implementing a planned,
well-tested, self-contained feature autonomously.

## What it is

A fresh `claude -p` process per task. Durable memory = the repo, not the chat.

```
        +-----------------------------------------------------+
        |  ralph.sh loop  (harness decides "done")            |
        |                                                     |
  +---> |  read tasks.md -> first [ ] task                    |
  |     |  fresh claude -p:                                   |
  |     |     search -> implement 1 task -> run GATES         |
  |     |     -> mark [x] -> git commit (no push) -> exit     |
  |     |  harness: progress? token? max-iter? stuck?         |
  +-----|  no -> next iteration (NEW context)                 |
        |  yes -> stop                                        |
        +-----------------------------------------------------+
                          |
                   human review + PR
```

Why fresh context: model quality rots past ~100-150k tokens (drift, forgotten
constraints, hallucination). Resetting every task keeps each run sharp; the
spec/plan/tasks files + git history carry the state.

## Decision: should I Ralph this?

```
 new/greenfield feature? ---- no ---> use /speckit-implement or Plan Mode
        | yes
 went through /speckit-plan + /speckit-tasks? -- no --> do that first
        | yes
 >= ~10 tasks AND real test gates? ---- no ---> /speckit-implement
        | yes
 files outside any whole-repo pre-commit check? -- no --> not safe; manual
        | yes
        v
   RALPH IT  (on a ralph/<feature> branch)
```

## Run it

```sh
git switch -c ralph/001-zscaler-icap-receiver
make ralph FEATURE=specs/001-zscaler-icap-receiver \
           GATES="go vet ./... && go test ./..." DRY_RUN=1   # validate first
make ralph FEATURE=specs/001-zscaler-icap-receiver \
           GATES="go vet ./... && go test ./..."             # real run
# review the branch, then open a PR
```

## Do / Don't

| Do | Don't |
|---|---|
| Use on new services/packages with a clean spec | Point it at the existing TS monorepo's web/worker |
| Make gates real (tests + lint + typecheck) | Trust the checkbox - gates are the proof of done |
| Start with `DRY_RUN=1` | Run on develop/main (the harness refuses) |
| Keep tasks 5-15 min each (at /speckit-tasks time) | Let it run unbounded - it is capped by max-iter |
| Review the branch + PR before merge | Auto-push (it never pushes; you do, after review) |
| Stop and refine tasks.md if it gets stuck (3 strikes) | Force a too-large task - it will blow context |

## Limits (be honest with the team)

- ~90% completion is the realistic ceiling; the last 10% (architecture, novel
  problems) is human.
- Greenfield bias is real. On legacy it only works on tightly-scoped, well-tested
  slices - otherwise it drifts.
- Cost scales with iterations x context. Bad specs -> "meh" output and wasted
  spend. Good `tasks.md` is the lever.

## Claude Code notes

- The harness uses the **external** `claude -p` loop on purpose (true fresh
  context). Do **not** use the in-session `ralph-wiggum` plugin for this - it
  keeps one context window, which is the thing we are avoiding.
- `--dangerously-skip-permissions` is required for unattended runs; that is why
  Ralph is confined to a `ralph/<feature>` branch and never pushes.
