#!/usr/bin/env bash
#
# ralph.sh - Fresh-context Ralph loop over a Spec Kit tasks.md work queue.
#
# Opt-in power tool for NEW / greenfield features only. Spec Kit
# (specify -> plan -> tasks -> analyze) is the standard for everything;
# this wraps the IMPLEMENT phase in a loop that runs a fresh `claude -p`
# process per task so context never rots.
#
# Each iteration: pick the first unchecked task in tasks.md, implement ONLY
# that task, run the quality gates, mark it [x], commit, and exit. The harness
# (not the model) decides "done".
#
# Guardrails (this script REFUSES to run otherwise):
#   - must run on a `ralph/<feature>` branch (never develop/main)
#   - feature must have gone through /speckit-plan + /speckit-tasks
#   - never pushes (you review the branch, then push/PR yourself)
#
# Usage:
#   .specify/ralph/ralph.sh [--feature DIR] [--gates "CMD"]
#                           [--max-iterations N] [--model NAME] [--dry-run]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FEATURE_DIR=""
GATES=""
MAX_ITER=""
MODEL="sonnet"
DRY_RUN=false

while [ $# -gt 0 ]; do
  case "$1" in
    --feature) FEATURE_DIR="$2"; shift 2 ;;
    --gates) GATES="$2"; shift 2 ;;
    --max-iterations) MAX_ITER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '[ralph] %s\n' "$*"; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

# --- Resolve feature dir (arg > SPECIFY_FEATURE > .specify/feature.json) ---
if [ -z "$FEATURE_DIR" ]; then
  if [ -n "${SPECIFY_FEATURE:-}" ]; then
    FEATURE_DIR="specs/${SPECIFY_FEATURE}"
  elif [ -f ".specify/feature.json" ]; then
    FEATURE_DIR="$(grep -oE '"feature_directory"[^,}]*' .specify/feature.json | sed -E 's/.*:\s*"([^"]+)".*/\1/')"
  fi
fi
[ -n "$FEATURE_DIR" ] || die "no feature dir (pass --feature specs/NNN-name or run /speckit-specify first)"
[ -d "$FEATURE_DIR" ] || die "feature dir not found: $FEATURE_DIR"

TASKS="$FEATURE_DIR/tasks.md"
PLAN="$FEATURE_DIR/plan.md"
FEATURE_NAME="$(basename "$FEATURE_DIR")"

# --- Spec-Kit-first guardrail ---
[ -f "$PLAN" ]  || die "$PLAN missing - run /speckit-plan before ralph"
[ -f "$TASKS" ] || die "$TASKS missing - run /speckit-tasks before ralph"
grep -qE '^- \[ \] T' "$TASKS" || die "no unchecked tasks in $TASKS - nothing to do"

# --- Branch guardrail: must be on ralph/<feature>, never develop/main ---
CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
WANT_BRANCH="ralph/${FEATURE_NAME}"
case "$CUR_BRANCH" in
  main|master|develop) die "refusing to run on '$CUR_BRANCH'. Create & switch to '$WANT_BRANCH' first: git switch -c $WANT_BRANCH" ;;
  ralph/*) : ;;
  *) die "not on a ralph branch (on '$CUR_BRANCH'). Expected '$WANT_BRANCH'. Switch with: git switch -c $WANT_BRANCH" ;;
esac

# --- Quality gates (backpressure). Override with --gates. ---
if [ -z "$GATES" ]; then
  if ls "$FEATURE_DIR"/../../*/go.mod >/dev/null 2>&1 || [ -f "${FEATURE_NAME}/go.mod" ]; then
    GATES="go vet ./... && go test ./..."
  else
    GATES="echo '[WARN] no gates configured - pass --gates \"<cmd>\"'"
  fi
fi

# --- Iteration bound: default = unchecked tasks + 10 ---
UNCHECKED="$(grep -cE '^- \[ \] T' "$TASKS" || echo 0)"
[ -n "$MAX_ITER" ] || MAX_ITER=$(( UNCHECKED + 10 ))

mkdir -p .specify/logs
RUN_PROMPT=".specify/ralph/.run-prompt.md"
sed -e "s#{{FEATURE_DIR}}#${FEATURE_DIR}#g" \
    -e "s#{{TASKS}}#${TASKS}#g" \
    -e "s#{{GATES}}#${GATES//#/\\#}#g" \
    .specify/ralph/ralph-prompt.md > "$RUN_PROMPT"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG=".specify/logs/ralph-${FEATURE_NAME}-${STAMP}.log"
ln -sf "$(basename "$LOG")" .specify/logs/ralph-latest.log

log "feature   : $FEATURE_NAME ($FEATURE_DIR)"
log "branch    : $CUR_BRANCH"
log "gates     : $GATES"
log "unchecked : $UNCHECKED task(s)"
log "max-iter  : $MAX_ITER"
log "model     : $MODEL"
log "log       : $LOG"
$DRY_RUN && { log "DRY RUN - validated guardrails + prompt, not invoking claude. Exiting."; exit 0; }

command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"

PREV_UNCHECKED="$UNCHECKED"
STRIKES=0
for i in $(seq 1 "$MAX_ITER"); do
  REMAIN="$(grep -cE '^- \[ \] T' "$TASKS" || echo 0)"
  if [ "$REMAIN" -eq 0 ]; then log "all tasks complete after $((i-1)) iteration(s)."; exit 0; fi
  log "=== iteration $i/$MAX_ITER ($REMAIN task(s) left) ==="

  # FRESH CONTEXT: new claude process each iteration; prompt from stdin.
  OUTPUT="$(claude -p --model "$MODEL" --dangerously-skip-permissions < "$RUN_PROMPT" 2>&1 | tee -a "$LOG")" || true

  grep -q '<promise>ALL_TASKS_COMPLETE</promise>' <<<"$OUTPUT" && { log "agent signalled completion."; exit 0; }

  # Progress / stuck detection: no task checked off this round counts as a strike.
  NOW_UNCHECKED="$(grep -cE '^- \[ \] T' "$TASKS" || echo 0)"
  if [ "$NOW_UNCHECKED" -ge "$PREV_UNCHECKED" ]; then
    STRIKES=$((STRIKES + 1))
    log "[WARN] no progress this iteration (strike $STRIKES/3)"
    [ "$STRIKES" -ge 3 ] && die "stuck: 3 iterations with no task completed. Review $LOG, then refine tasks.md or run /speckit-analyze."
  else
    STRIKES=0
  fi
  PREV_UNCHECKED="$NOW_UNCHECKED"
  sleep 1
done

log "hit max-iterations ($MAX_ITER) with $(grep -cE '^- \[ \] T' "$TASKS") task(s) left. Review $LOG."
exit 1
