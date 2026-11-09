from backend.realtime_provider import DEFAULT_ADA_INSTRUCTIONS


def test_card_status_guidance_uses_existing_compact_surfaces():
    assert "use nest_status once for DevNest overview" in DEFAULT_ADA_INSTRUCTIONS
    assert "use kanban-brief then named-card reads for cards" in DEFAULT_ADA_INSTRUCTIONS
    assert "reuse answers and don't full-scan/repoll" in DEFAULT_ADA_INSTRUCTIONS
