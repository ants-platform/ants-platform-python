# Spec-Driven Development - team guide

How we build at Agentic Ants. Spec Kit is the standard for every change. Ralph is
an opt-in loop for new features. Read once; keep it open for the diagrams.

---

## The pipeline

```
 constitution  (once per repo, governs everything)
      |
      v
 /speckit-specify   --> spec.md      the WHAT / WHY  (no tech)
      |
 /speckit-clarify   --> spec.md      kills ambiguity (<=5 Qs)
      |
 /speckit-plan      --> plan.md      the HOW (stack, design, contracts)
      |
 /speckit-tasks     --> tasks.md     dependency-ordered checklist (the queue)
      |
 /speckit-analyze   --> report       cross-check spec/plan/tasks (read-only)
      |
 /speckit-implement --> code         build it (or: Ralph loop for greenfield)
```

Artifacts live in `specs/NNN-feature/`. Each is reviewed before the next step.

---

## Install (once per machine)

```sh
# 1. Spec Kit CLI (uv-based)
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
specify --version

# 2. Per repo - already done for our repos; only for a brand-new repo:
specify init --here --integration claude --script sh --branch-numbering sequential
```

Our 9 repos already have `.specify/` + the constitution + tailored templates
committed. You only run `specify init` when creating a *new* repo.

```sh
# 3. Ralph loop (optional, already committed) - needs the Claude Code CLI on PATH
claude --version
```

---

## Day-to-day flow

```
   idea / ticket
        |
        v
   /speckit-specify  ->  review spec.md   --- unclear? --> /speckit-clarify
        |  ok
        v
   /speckit-plan     ->  review plan.md   (Constitution Check must pass)
        |  ok
        v
   /speckit-tasks    ->  review tasks.md
        |
        v
   /speckit-analyze  ->  fix CRITICAL/HIGH findings
        |
        +--- greenfield + tested + >=10 tasks? --> Ralph loop (own branch)
        |
        +--- everything else --------------------> /speckit-implement
        |
        v
   tests pass -> PR -> review -> merge
```

One feature = one branch = one `specs/NNN-feature/` dir.

---

## Ralph loop (opt-in, greenfield only)

```sh
git switch -c ralph/NNN-feature
make ralph FEATURE=specs/NNN-feature GATES="<test+lint cmd>" DRY_RUN=1   # check
make ralph FEATURE=specs/NNN-feature GATES="<test+lint cmd>"            # run
# review branch -> PR
```

Fresh `claude -p` per task: implement one -> gates pass -> mark `[x]` -> commit ->
exit -> repeat. Bounded by max-iterations; never pushes. Full details:
[`ralph-loop.md`](./ralph-loop.md).

---

## Do

```
[+] Hand-write/curate the constitution; let it gate every PR
[+] Keep one feature per branch, scope bounded (one capability)
[+] Review EVERY generated artifact - the agent can silently ignore the spec
[+] Run /speckit-clarify when scope is fuzzy (cheaper than rework)
[+] Tests are the proof of done - constitution mandates Test-First
[+] Commit per logical step; let the pre-commit hooks run (no --no-verify)
[+] Re-run a command to iterate; don't hand-edit downstream artifacts
```

## Don't

```
[x] Hand-edit tasks.md/plan.md between steps (regeneration clobbers it)
    -> only the constitution is yours to hand-edit
[x] Use Ralph on the TS monorepo's web/worker (greenfield only)
[x] Skip /speckit-analyze on anything non-trivial
[x] Push Ralph branches without a human PR review
[x] --no-verify / bypass pre-commit hooks (constitution forbids it)
[x] Use the in-session ralph-wiggum plugin (context accumulates - defeats it)
[x] Add UTF-8 emojis anywhere (use [OK]/[ERROR]/[FAIL])
```

---

## When NOT to use the full pipeline

```
 tiny bug fix / 1-file change ........ just Plan Mode or /speckit-implement
 < ~10 tasks ......................... specify -> plan -> implement (skip Ralph)
 exploratory / unknown end-state ..... prototype first, spec after
 no test gates ....................... add tests before you automate
```

The ceremony pays off on sizeable, well-understood features. For small work it is
overhead - use judgment.

---

## Cheat sheet

| Command | Output | Edit by hand? |
|---|---|---|
| `/speckit-constitution` | `.specify/memory/constitution.md` | YES (the only one) |
| `/speckit-specify` | `spec.md` + branch | no - re-run to change |
| `/speckit-clarify` | spec.md updates | no |
| `/speckit-plan` | `plan.md`, `research.md`, `data-model.md`, `contracts/` | no |
| `/speckit-tasks` | `tasks.md` | no |
| `/speckit-analyze` | report (read-only) | n/a |
| `/speckit-implement` | code | n/a |
| `make ralph ...` | code (loop, greenfield) | n/a |

Questions -> see `ralph-loop.md` and each repo's `.specify/memory/constitution.md`.
