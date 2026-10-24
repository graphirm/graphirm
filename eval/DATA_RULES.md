# Evaluation data rules

These rules apply to every remaining run and to future evals. The evaluation
is an audit trail, same as the product.

## 1. One registry of everything

`eval/registry.json` is the single source of truth: tracks, tasks (IDs,
exclusions and why), arms, configs, model and provider, prices, budgets,
binaries and dataset versions. Each entry has a sha256 and a git commit.

Analysis scripts read only from the registry and the run ledgers. They do not
copy counts, prices, or task lists by hand.

Task IDs live in the file the registry names. A script loads that file only
after its sha256 matches the registry. Exclusions are listed in the registry
itself, with the reason.

## 2. Build only from committed code

Commit before any build used in a run. `eval/build_binary.py` refuses to build
an eval binary from a dirty tree.

`build.rs` embeds the git commit and a dirty flag. `graphirm --version` prints
them. `run_metadata` and the run summary include `git_commit` and `git_dirty`.

The build command, toolchain version and output sha256 go in the registry
before the binary is used.

## 3. Immutable, verified run outputs

Ledgers are append-only. A correction is a new row that names the old row.
Rows are not edited.

Each completed run writes `<arm>.manifest.json`: the output files and their
sha256 (patch, summary, grade, harness report, graph). `eval/swebench/integrity.py`
writes that manifest. A later run writes a new manifest name. It does not
replace the old one.

After every track, `finish_track` checks that every ledger row has its files,
every recorded hash matches, and grades match the harness. The track counts as
complete only when that check prints no problems. The drivers do not clear the
`running` file until then.

## 4. Backups (3-2-1)

Three copies, on two kinds of storage, one off this machine.

- Local durable disk: `/home/krs/graphirm-runs/<track>/`.
- The GitHub backup repo for small files and the binaries.
- Off-machine storage for graphs at the end of a track.

Verify a backup by cloning it fresh and checking hashes. Run `git fsck` on
backup repos after each track. Outputs do not live only in `/tmp`.

## 5. One definition per metric

`eval/swebench/metrics.py` defines resolved, primary cost, billed cost, cache
hit and tokens once, and names the source field. Every new analysis imports it.
There is no per-arm formula.

A change to a definition is a journal entry. Earlier results are recomputed
with the new definition. Both versions are kept.

## 6. Decisions are logged before they take effect

Deviations, exclusions, cap changes and reruns are journalled with a timestamp
before the change. The registry's `decisions` list links each journal entry to
the rows it affects.

## 7. What a backfill may say

A fact that was not recorded at the time is `unknown`, with the reason. It is
not filled in from memory.
