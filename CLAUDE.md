# Guidelines

* **Package management:** Use uv, not pip. Installation: `uv add <packagename>`, running code: `uv run ...`, etc.
* **Code quality:** Use type hints in all public-facing methods, including `jaxtyping` for array dimensionalities. Functions must be focused and small, and all public-facing functions should have full docstrings.
* **Philosophy:** Write simple, straightforward code. Make it easy to understand. Consider performance, but don't sacrifice readability. Less code = less debt -- minimize code footprint!
* **Environment:** You are in a SLURM cluster environment, running on a head node. If you need to do heavy compute, *ALWAYS* launch a job with `srun`, even if it's just a CPU job. Always make sure to specify the number of cpus (e.g. `-c 6`) and memory (eg. `--mem=32G`). No more than 32 CPUs or 128G of RAM should be assumed readily available. If you really need a GPU, use `--gres=gpu:1` along with `--partition=gpu100` for an H100 (or `--partigion=gpu` for a smaller GPU). The `debug_gpu` partition can also be used for short-term, single-GPU debugging, sometimes with shorter wait times (and no H100s).
* **Breaking changes are OK:** This is a self-contained experimental codebase for the time being. Breaking changes, especially in the interest of reducing the amount of code, are perfectly acceptable. Do not deprecate unused code/branches -- delete them altogether.
* **Unit tests:** Write human-readable unit tests for each class/module you develop.
* **Overview:** Maintain an accurate overview of all files in the project in `OVERVIEW.md` with a one-line description. Use this file to minimize looking at files. Only look at a file if you definitely need the context.
* **Experiment guide:** In each experimental config file in `conf`, maintain an up-to-date, brief, technical description of the experiment. Use math if needed. Make sure it accurately reflects the experiment end-to-end.
