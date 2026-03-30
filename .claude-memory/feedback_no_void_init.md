---
name: No void initialization
description: Never use void initialization in D code — always zero-initialize arrays and variables
type: feedback
---

Never use `= void` initialization in D code. Always zero-initialize.

**Why:** void initialization leaves garbage values that cause subtle bugs (e.g., Gram-Schmidt failing because of uninitialized memory). The performance benefit is negligible compared to the debugging cost.

**How to apply:** Use `= 0` for numeric arrays, default init for structs. Never write `double[4] v = void`.
