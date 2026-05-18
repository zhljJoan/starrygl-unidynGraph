# Development Environment Notes

## Codex GPU Access

In this project, Codex commands run in a restricted sandbox by default. The
default sandbox can execute normal CPU/file-system commands, but it may not see
GPU device nodes such as `/dev/nvidia*`.

Observed on 2026-05-18:

- Default sandbox:
  - `ls -l /dev/nvidia*` produced no visible devices.
  - `nvidia-smi` failed with: `couldn't communicate with the NVIDIA driver`.
  - The effective groups were `zlj nogroup`.
- Escalated command execution:
  - `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader`
    reported four NVIDIA A40 GPUs.

Conclusion: a default-sandbox `nvidia-smi` failure in Codex does not mean the
machine GPU driver is broken. It means the current Codex command sandbox does
not expose GPU devices.

For GPU-dependent validation, run the command with escalated permissions, for
example:

```text
nvidia-smi ...
python ...  # when the script initializes CUDA
torchrun --nproc_per_node=4 ...
```

When reporting test status, distinguish these two cases:

- "GPU unavailable in default Codex sandbox" means the sandbox lacks device
  access.
- "GPU/driver unavailable on host" should only be reported after an escalated
  command also fails.
