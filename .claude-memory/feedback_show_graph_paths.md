---
name: feedback_show_graph_paths
description: Always print file paths of generated graphs so user can open them
type: feedback
---

When creating graphs/plots, always print the file path clearly so the user can open them (e.g. in a file manager or with xdg-open). Don't assume the user can see images inline in the chat.

**Why:** The user cannot see images displayed inline in Claude Code. They need the file path to open the image themselves.

**How to apply:** After saving any plot, print a clear message like "Graph saved to: /tmp/foo.png" so the user can navigate to it.
