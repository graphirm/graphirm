# eval/

Drivers, grading wrappers and analysis scripts for Graphirm's SWE-bench and multi-session evals.
`swebench/PLAN.md` and `swebench/prereg.md` hold the pre-registered tracks. Some drivers and the
frozen-binary registry still live on the eval branches; they reach `main` under a task.

## Which eval code needs a plan

Work is tracked with epicure (see the root `AGENTS.md`). For eval code:

- **Exploratory code is a spike.** A probe, a one-off driver or a checkpoint run is filed as a spike
  item. It may live in a worktree or scratch space, and nothing on `main` may depend on it.
- **Code a recorded run depends on is a task under an approved eval plan.** That is anything a binary registry
  names, and any script whose output a plan's results quote. It is committed before the run
  it serves, so the run can name the commit.
- **Pre-registered limits come first.** A pre-registered plan states its pass limits before its first
  task runs, and they are not changed after.
- **Run outputs stay out of the repo.** Ledgers, grader logs, patches and summaries go under
  `~/graphirm-runs/`, never in this repo or the work repo. Plans quote the numbers and name the paths.

Example: a driver for a new arm (such as `run_pieces_wording.py`) is a task under that arm's plan,
filed and committed before its checkpoint runs.

## Before a run

- Build the eval binary from a committed, clean source, and record its hash and source commit.
- Check the track's integrity before drawing any verdict.
