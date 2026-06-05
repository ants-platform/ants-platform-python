# Ralph Build Mode - one task per run, then exit

You are running in a fresh-context loop. You have NO memory of prior iterations.
Durable state lives ONLY in the repo: the spec, the plan, tasks.md, and git history.
Re-derive "where am I" from those files every time. Do ONE task, then stop.

## Feature

- Spec:  {{FEATURE_DIR}}/spec.md
- Plan:  {{FEATURE_DIR}}/plan.md
- Tasks: {{TASKS}}   <- the work queue (checkbox list)

## Procedure (do exactly this, once)

0a. Read {{TASKS}} and find the FIRST unchecked task `- [ ] T###`.
0b. Read the relevant parts of spec.md, plan.md, data-model.md, contracts/.
0c. SEARCH THE CODEBASE FIRST. Do not assume the task is unimplemented -
    grep for the types/functions/files it describes before writing anything.
1.  Implement ONLY that one task. Keep changes minimal and follow existing
    patterns. Honor `.specify/memory/constitution.md` (Test-First, no UTF-8
    emojis, type safety, prompt-content rules, etc.).
2.  Run the quality gates - they MUST pass before you continue:
        {{GATES}}
    If they fail, fix your change until they pass. If the task turns out to be
    much larger/more complex than expected, STOP without committing and explain
    why (do not blow context forcing it).
3.  Mark the task `[x]` in {{TASKS}} (only that one task).
4.  Commit just this task's changes:
        git add -A && git commit -m "feat(<area>): T### <short summary>"
    Do NOT push. Do NOT switch branches.
5.  Exit. The harness starts the next iteration with a fresh context.

If, in step 0a, there are NO unchecked tasks left, output exactly:
<promise>ALL_TASKS_COMPLETE</promise>
and do nothing else.

## Rules

- ONE task per run. After committing one task, stop - do not start another.
- Tests/gates are the proof of done, not the checkbox. Never mark `[x]` on a
  task whose gates do not pass.
- No placeholder/stub code, no `TODO` left behind, no fake implementations.
- Search before you write (the #1 failure mode is re-implementing existing code).
- Delegate expensive reads to subagents to keep your context lean; do the build
  yourself.
- Never edit develop/main; never push; never run destructive DB or infra commands.
- Stay strictly inside this feature's scope (the spec). No emergent extras.
