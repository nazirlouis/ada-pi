from backend.ada_opencode_control import preflight


def test_opencode_preflight_requires_explicit_envelope_and_budget():
    result = preflight(
        {"id": "ada-card", "ref": "ADA-1010", "updated": "now",
         "action": {"executor": "opencode"}},
        {"repo": "chaba", "model": "litellm/gemini-3.5-flash-lite"},
    )

    assert result["ok"] is False
    assert result["ready_for_dispatch"] is False
    assert "llm_key_alias" in result["missing"]
    assert "max_usd" in result["missing"]
    assert result["mutations"] == []


def test_opencode_preflight_accepts_complete_explicit_envelope():
    result = preflight(
        {"id": "ada-card", "ref": "ADA-1010", "updated": "now",
         "action": {"executor": "opencode"}},
        {
            "repo": "chaba",
            "model": "litellm/gemini-3.5-flash-lite",
            "llm_key_alias": "bench",
            "profile": "ops-readonly",
            "target_host": "idc02",
            "artifact": "artifacts/ada-card/result.json",
            "max_usd": 0.25,
            "max_turns": 3,
            "max_tokens": 12000,
            "max_wall_time_s": 600,
            "max_retries": 0,
        },
    )

    assert result["ok"] is True
    assert result["ready_for_dispatch"] is True
    assert result["approval_required"] is True
    assert result["mutations"] == []
