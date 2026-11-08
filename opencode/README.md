# ada-code deploy (aka opencode-ada)

**Naming (2026-11-08):** `ada-code` is the canonical name for Ada's OpenCode
responder — the surface where Ada answers via OpenCode + ada-control MCP.
`opencode-ada` / `ada-opencode` are the SAME thing (legacy names still used
by the hostname, quadlets, and volume dirs — treat as aliases, do not create
a second instance).

Canonical sources for the ada-code OpenCode instance (container on idc02,
serve :4096, UI at ada-ha.surf-thailand.com/chaba-admin/opencode-ada).

Deploy path: ~/.local/share/ada-code/workspace/
  opencode.json            -> workspace/opencode.json
  ada_mcp.py               -> workspace/.opencode/ada_mcp.py
  commands/ada.md          -> workspace/.opencode/commands/ada.md

Edit here, copy to the workspace, then `systemctl --user restart ada-code`.
