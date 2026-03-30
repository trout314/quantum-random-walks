---
name: D toolchain location
description: ldc2 compiler and D toolchain installed at ~/dlang/ldc-1.41.0; not on default PATH — must use full path or source ~/dlang/ldc-1.41.0/activate
type: reference
---

D compiler (ldc2) is installed at `/home/aaron-trout/dlang/ldc-1.41.0/bin/ldc2`.
Bash completion script at `/home/aaron-trout/dlang/ldc-1.41.0/etc/bash_completion.d/ldc2`.
The build system is meson + ninja, configured in `dlang/build/`.
To build: ensure ldc2 is on PATH (or the meson build dir was configured with the right compiler path), then `ninja -C dlang/build`.
