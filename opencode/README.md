# ada-opencode deploy

Canonical sources for the ada-opencode OpenCode instance (container on idc02,
serve :4096, UI at ada-ha.surf-thailand.com/chaba-admin/opencode-ada).

Deploy path: ~/.local/share/ada-opencode/workspace/
  opencode.json            -> workspace/opencode.json
  ada_mcp.py               -> workspace/.opencode/ada_mcp.py
  commands/ada.md          -> workspace/.opencode/commands/ada.md

Edit here, copy to the workspace, then `systemctl --user restart ada-opencode`.
