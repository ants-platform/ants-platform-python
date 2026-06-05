# .specify/ralph - fresh-context Ralph loop (opt-in)

Wraps the Spec Kit **implement** phase in a loop that runs a fresh `claude -p`
process per task, so context never rots. **New/greenfield features only.**
Spec Kit (specify -> plan -> tasks -> analyze) stays the standard for everything.

## Files

| File | Purpose |
|---|---|
| `ralph.sh` | the loop harness (guardrails + iteration + stop conditions) |
| `ralph-prompt.md` | the per-iteration build prompt (one task, gates, commit, exit) |
| `.run-prompt.md` | generated per run from the template (gitignored) |

## Use

```sh
git switch -c ralph/001-zscaler-icap-receiver        # required: ralph/<feature> branch
make ralph FEATURE=specs/001-zscaler-icap-receiver GATES="go vet ./... && go test ./..."
# or directly:
.specify/ralph/ralph.sh --feature specs/001-zscaler-icap-receiver \
                        --gates "go vet ./... && go test ./..." --dry-run
```

Always start with `--dry-run` to confirm guardrails + the resolved prompt, then
drop it for the real run. Review the branch and open a PR yourself - ralph never
pushes.

See `docs/Development_doc/ralph-loop.md` for when to use / when NOT, and
`docs/Development_doc/spec-kit-developer-guide.md` for the team standard.
