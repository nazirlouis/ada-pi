# Dispatch outcome — voice-dispatch-card-autocreate

**Card:** voice-dispatch-card-autocreate · **Runner:** idc02 ·
**Branch:** `dispatch/20261109-181807-repo-ada-voice-primary-change-`
(commit `4566363`, not pushed)

## What changed

`devin action='dispatch'` with no playbook now files an armed kanban
card via board-api instead of ssh'ing `devin-dispatch start` on
tony-dell — every voice/chat dispatch is board-tracked and claimed by
kanban-dispatch.

- `backend/devin_dispatch.py`
  - new `_board_dispatch()` — POST `/card` via the shared
    `backend/board_client.post` (the single board reach path;
    `ADA_BOARD_API_URL` env override, default
    `https://tony-dell.taila0626a.ts.net/apps/board-api`):
    `{id: voice-dispatch-<epoch>, title: task[:80], spec: task +
    "Dispatched by Ada voice/chat on behalf of the user.", column:
    backlog, tags: [voice-dispatch], priority: medium, action: {type:
    dispatch, repo}, queue: true, on_exists: queue, from: ada, text:
    provenance}`
  - unreachable board (status 0) → `{ok: False, error}` — **no** ssh
    fallback (per card: the tuned path is the whole point); whitelist
    400s surface the board-api `error` verbatim
  - success → `{ok: True, card_id, note: "queued on the board —
    kanban-dispatch will claim and run it"}`; card_id parsed from the
    response (structured fields then `card <id>` message text, same as
    `tools.d/kanban.py` `_filed_card_id`), deterministic id as fallback
  - dedup extended: entries carry `is_card`; card dupes return
    `{ok, card_id, deduplicated}` — playbook/session dupes still return
    `task_id`; `_remote_dupe` unchanged (a claimed card runs as a
    devin-dispatch session anyway)
  - playbook path, `status`/`followup`/`tasks`, gates, and the
    job-contract stamp are untouched
- `backend/realtime_provider.py` `DEVIN_INSTRUCTIONS` — report the CARD
  id and that the board queues/runs it (task id for playbooks);
  dispatch is asynchronous, watcher notifies on completion as before
- `backend/tool_guide.yml` `devin:` blurb — same updated wording
- `tests/test_devin_dispatch.py` — new `BoardDispatchTests` (payload
  verbatim incl. queue:true, title truncation, result shape, id parse,
  unreachable-without-ssh-fallback, 400 verbatim); dedup tests reworked
  for the card path; 33 tests
- `docs/ssot/jobs/ada/2026-11-09-voice-dispatch-board-card.yml` — SSOT
  trail

## Result

- `python3 scripts/tool-lint.py` — clean
- `pytest tests/ -x -q` — **1739 passed, 7 skipped** (ran with a
  worktree-local pytest + the sibling checkout's dep venv via
  PYTHONPATH, since this worktree ships no venv; `.venv-test` removed
  after)

## How to verify

- Unit: `pytest tests/test_devin_dispatch.py` — payload asserted
  verbatim against a mocked `board_client.post`
- Post-deploy e2e (deferred — deploy only via `deploy-ada.sh` on idc03
  per AGENTS.md; a live `queue:true` card spawns a real session so it
  was not triggered from this session): on the ada-dev lane ask Ada to
  dispatch a task → a `voice-dispatch-*` card lands queued on the
  board → kanban-dispatch claims it → card comms show "primer
  injected"; the watcher notifies on completion as before.
