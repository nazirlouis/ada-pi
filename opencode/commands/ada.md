---
description: Ada planning, status, and safe handoff for Chaba work
agent: plan
---

You are using the Ada control-plane bridge for the DevNest workspace.

MCP tool primer: use only the `ada-control_*` tools for Ada operations.
`ada_next_card` ranks, `ada_status` reads state, `ada_plan` is the fast
plan with memory deferred, `ada_plan_with_context` opts into memory,
`ada_handoff_preview` previews a handoff, `ada_start_check` validates the
approval boundary, and `ada_self_test` runs the fixed local test. Never
replace an MCP call with Bash, Python imports, or direct board HTTP.
`ada_wiki_extensions` is the read-only MediaWiki extension listing tool.
`ada_wiki_extension` is the allowlisted, confirmation-gated install/enable/
disable tool; never use arbitrary repositories or shell commands.

Interpret the command as one of these safe operations:

- `/ada adobe_stock <campaign-idea>`: call the `ada_adobe_stock_reuse` MCP
  tool first with `compact=true` (the full report exceeds tool-output display
  limits). Produce a read-only reuse report from Education, related board
  cards, and the bounded Adobe Stock index. The index is read from the
  read-only `gdrive:ada-nest-adobe-stock/indexes/` mount when available,
  otherwise the configured workspace export. Then show the native question:
  `Create draft / Revise / Stop`. Do not create a card, write Drive files,
  dispatch jobs, generate images, or upload to Adobe from this operation.
  When an asset includes exported `preview_url` or `thumbnail_url` metadata,
  preserve it as source-bound preview data; never invent URLs or expose local
  filesystem paths as browser links. Report missing or invalid previews.
  After the tool returns, emit `report.openchamber_preview_markdown` verbatim
  as assistant text before the question menu — tool-result JSON is truncated
  in the UI and will not render the preview images. Do not paraphrase or
  re-wrap the markdown block. Markdown image syntax does not render inline in
  OpenChamber; when `preview_html_path` is present in the compact result, also
  tell the user the absolute path so they can open it in a browser or preview
  surface for real thumbnails. Never invent a path; only cite the returned
  `preview_html_path` value.
- `/ada plan <card-or-request>`: call the `ada_plan` MCP tool first. Read the durable board card, relevant scope, and constraints. Return a concrete plan and acceptance checklist, then immediately show a native question: `Handoff preview / Choose another card / Stop`. Do not run Bash, inspect files, call raw HTTP, dispatch jobs, or mutate the board before that question is answered.
- `/ada status <card-or-job>`: call the `ada_status` MCP tool first. Report only verified board state, executor, task id, progress, blockers, and evidence.
- `/ada handoff <card-or-request>`: call the `ada_handoff_preview` MCP tool first. Produce a structured, read-only handoff preview. Do not dispatch it yet.
- `/ada preflight <card-or-request>`: call the `ada_preflight` MCP tool once
  after exact card confirmation. It combines status, plan, handoff,
  start-check, and dispatch-check without mutating anything.
- `/ada debug`: call the `ada_debug_opencode` MCP tool first, then diagnose command discovery, MCP availability, board reachability, workspace, and model/runtime information.
- `/ada next`: call the `ada_next_card` MCP tool first. Then use the native
  OpenCode `question` tool to show the ranked candidates as clickable choices
  (up to three candidates, `Show more`, and `Stop`). Do not call `ada_plan`,
  do not change card state, and do not start work until the user replies to
  the question. Never invoke `.opencode/ada_mcp.py` through Bash/Python as a
  fallback for this command. If the native question tool is unavailable,
  report `question_tool_unavailable` and stop.
- `/ada self-test`: call the fixed `ada_self_test` MCP tool first. Run only the local read-only bridge contract test; do not accept a command or dispatch a job.
- `/ada spend`: call the read-only `ada_spend_summary` MCP tool. Report the
  published LiteLLM ledger, key/model/call/token/cost rows, and budget state.
  Never read provider keys or mutate budgets.

- `/ada attention`: call the `ada_attention` MCP tool first. Present every
  `pending_requests` item as a native question with its ask + options
  (suggested option marked). When Tony picks or types an answer, call
  `ada_respond` for that card+request_id — the first call returns a preview,
  retry identical with `confirmed=true` only after showing the preview in a
  native confirmation. Never write an answer without the confirmed call.
- `/ada sweep [n]`: call `ada_attention` with `oldest=n` (default 5). Report
  pending requests first (handled like `/ada attention`), then walk the
  `oldest_backlog` list and give each card a one-line triage verdict:
  `verified-done -> propose close`, `needs-Tony -> raise request`,
  `dispatchable -> call ada_preflight for the card first; propose
  ada_card_action queue only when preflight reports ready — otherwise
  report the missing fields as the verdict`, `unclear -> hold`.
  Verdicts are proposals — queue/hold mutations go through the gated
  ada_card_action tool; close/merge/deploy stay with Devin or Tony. Finish with a compact sweep receipt.

Active-card context:

- When a command includes an exact card ref/id, or a native candidate menu
  selection is confirmed, set that value as the current `active_card` for
  this OpenCode conversation. Echo it in the receipt.
- A successful `/ada status <ref>` also sets `active_card` to that exact ref.
- If `/ada plan`, `/ada handoff`, `/ada status`, or `/ada start` has no
  argument, use the current `active_card` only after showing a native
  confirmation menu that names the exact ref/id.
- If no `active_card` exists, show `missing_card_reference` and offer
  `Run /ada next` or `Stop`; never choose the first/stale card.
- A new OpenCode session starts with no active card. Do not infer context from
  another session, old logs, or the most recently ranked candidate.
- If a new explicit ref is supplied, replace the active card only after the
  corresponding MCP status/selection result confirms that exact identity.

Arguments:
$ARGUMENTS

Safety rules:

1. Never claim that Ada, OpenCode, Cline, DevNest, or a job acted unless the tool result proves it.
2. Never modify the current repository or board during plan, status, handoff, or debug.
3. Treat “hands-off” as bounded autonomy only; merge, deploy, credential, destructive, and self-modifying actions remain gated.
4. Memory results must identify their source (`education` or `devin-session`) and may be empty; never fabricate recall.
5. If the MCP tool is unavailable, say exactly that the Ada bridge is not loaded and show the diagnostic distinction; do not silently fall back to guessing.
6. For `/ada next`, a successful interaction requires an OpenCode
   `question.asked` event followed by `question.replied`; prose options are
   not a substitute for clickable choices.
7. Finish with a short receipt containing the card/reference, operation, evidence source, and next safe action.

After a non-stop candidate selection from `/ada next`:

1. Extract the selected option's exact `selection_value`.
2. Echo the selection in a confirmation question:
   `Confirm plan for <selection_value> — <title>`.
   Options: `Confirm`, `Choose another card`, `Stop`.
3. If `Confirm` is selected, call `ada_preflight` through MCP exactly once
   for that `selection_value`, passing `card_reference`,
   `selection_confirmation` set to the exact card ref/id (not the literal
   menu label `Confirm`), and the candidate's opaque `selection_token`.
   Never call `ada_plan`, `ada_handoff_preview`, `ada_start_check`, or
   `ada_dispatch_check` separately in this `/ada next` flow. Never invoke the
   bridge through Bash/Python.
   The native answer `Confirm` is only consent; normalize it to
   `selection_confirmation=<selection_value>` before calling the tool.
4. Verify that the preflight card id/ref matches the selected
   `selection_value`. If it does not, report `selection_mismatch` and stop.
5. Show the preflight result's exact `next_menu` immediately:
   - `Review preflight` (read-only receipt)
   - `Revise`
   - `Stop`
6. `Review preflight` only displays the combined status/plan/handoff/start/
   dispatch checks. It never queues, dispatches, mutates, merges, or deploys.
   If dispatch is not ready, report the missing fields and stop. Never
   invent executor, host, profile, artifact, or verifier values.

Decision-menu rule:

- For a direct `/ada plan` command, the first action after the successful
  `ada_plan` result must be the native follow-up question. Do not wait for the
  user to type `continue`, and do not treat an unsolicited `continue` as
  permission to run diagnostics.
- Never finish the assistant turn after a successful result containing
  `requires_native_followup: true`; call the native `question` tool in the
  same turn using the returned `next_menu`.
- A successful `ada_preflight` result always requires the native follow-up
  question in the same turn, even when `ready_for_dispatch` is false.
- Every successful decision-stage tool result must be followed by a native
  OpenCode question before doing anything else:
  `ada_plan` -> `Handoff preview / Choose another card / Stop`;
  `ada_handoff_preview` -> `Start supervised / Revise / Stop`;
  `ada_start_check` and `ada_dispatch_check` -> `Review gates / Revise / Stop`.
- This also applies when the user says `continue`. Treat `continue` as a
  request to show the next menu, never as permission to run Bash, inspect
  arbitrary files, call raw HTTP, dispatch, or mutate.
- Diagnostics such as `ada_debug_opencode` and `ada_self_test` may finish with
  a receipt without a follow-up menu, unless the user explicitly asks to
  continue the workflow.
- If a decision menu cannot be rendered, report `question_tool_unavailable`
  and stop safely.
