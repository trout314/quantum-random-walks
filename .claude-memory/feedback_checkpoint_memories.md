---
name: Save checkpoint memories before long runs
description: Save a project memory at meaningful checkpoints so sessions can resume after crashes
type: feedback
---

Save a project memory at every meaningful checkpoint: after identifying a bug, before kicking off a long computation, after a test run produces interesting/concerning results.

**Why:** The computer froze from OOM thrashing during a sigma=30 walk run, killing the session. Recovering context required manually reading JSONL session logs.

**How to apply:** At each checkpoint, write/update a `project_current_thread.md` memory capturing: what we just observed, what the open question is, and what the next step would be. This makes "resume last session" trivial.
