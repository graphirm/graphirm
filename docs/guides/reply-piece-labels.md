# Labeling reply pieces

A final Pi reply is split into pieces. You name the shape of each piece. You do not use the question, and you do not mark whether the reply is correct.

Label final replies only. Narration before a tool call is out of this set.

## Cuts

One piece has one kind. If a piece seems to be two kinds, the parser cut it wrong. Split it. Do not give it two labels. A piece the labeler cannot name with one word is a failed cut, not a fuzzy choice.

Cut by function, not by sentence. "Plain text, or `nb`, or `jrnl`" is one options piece inside one sentence. A three-sentence explanation can be one statement.

How something works is a statement. What the reader does is steps. "Spark reads the file, shuffles, then writes" is a statement. It is in the labeler's test set because the likely mistake is to call it steps. It is not a new kind.

## Kinds

**statement** — A claim or a fact, with no list of choices and no warning.

> The server is already running on port 3000.

Near-miss: "This will print `hi`." is a statement. It describes what the code does. It is not a caveat.

**options** — A list of alternatives. The reader picks one. Order does not matter.

> - Use the cache
> - Skip the cache

Near-miss: "1. Stop the process. 2. Start it again." is steps. The order is the procedure.

**steps** — A list the reader follows in order.

> 1. Stop the old process.
> 2. Start it again with the same env.

Near-miss: "Use X / Use Y" is options, even though each line is an imperative.

Near-miss: "Spark reads the file, shuffles, then writes" is a statement. It says how the process works. The reader is not being told to do those things.

**instructions** — Directions for how to do something, written as prose or as a list of directions, where the reader is being told how to act but the lines are not a sequence of commands and not a menu of choices.

> Pass the token in the `Authorization` header. Keep the secret out of the query string.

Near-miss: a numbered restart procedure is steps.

**example** — A sample shown to illustrate, not the change under discussion.

> For example, a request looks like `GET /health`.

Near-miss: a rust fence that is the patch itself is code.

**caveat** — A warning about a cost, a limit, or a consequence of the thing just shown.

> This will invalidate tokens issued under the old bug.

Near-miss: "This will print `hi`." is a statement.

**code** — A snippet that is the change, the command, or the code under discussion.

```rust
if claims.exp * 1000 < now_ms() {
```

Near-miss: the same fence, introduced as a sample of what not to ship, is an example.

**question** — The reply asks the reader something.

> Should I apply the patch?

Near-miss: "Let me know if you want me to push it." has no question mark. Label it as a question only when it is asking for a decision. Otherwise it is a statement.

## What you are labeling

- `order` is the piece number. Item numbers inside a list are a separate count.
- A heading is the line the reply wrote, or none.
- A list is one piece. Its lines stay lines.
- If the tool says coverage failed, you are looking at the whole reply as one block.

## How to label

```bash
graphirm label-pieces ~/.graphirm/pi-runs
```

Type one kind name per piece: `statement`, `options`, `steps`, `instructions`, `example`, `caveat`, `code`, or `question`. Type `quit` to stop. Finished replies are appended to `piece-labels.jsonl`. The next run reads that file and skips a reply that is already there. A reply you quit in the middle of is asked again.

Each row stores the run file, the segment index inside that file, a SHA-256 of the segment text, the parser version, the baseline version, and each piece's byte range. It does not store "piece 3". After a parser change, the scorer can tell which ranges still match and which blocks moved.

```bash
graphirm score-pieces piece-labels.jsonl
```

The score is split into single-piece replies and replies with several pieces. A single statement such as "The file content is: `hi`" does not carry the several-piece score. Use Pi on varied tasks so lists, fences, and questions are in the set.

The first pass does not show a suggested kind. Do not pass `--show-baseline` until a blind portion is done. Accepting a suggestion makes the baseline look better than it is.

Recordings start only after the server is built from `feat/reply-pieces`. The files in `~/.graphirm/pi-runs` contain thinking, tool arguments, and file contents. They are not committed.

## HTML index

A reply Pi writes as an HTML index is cut by `cut_html_index`, not by `structure_segment`. The kind on each part is the class Pi wrote. The class is one of the eleven kind words and nothing else: statement, recommendation, options, steps, instructions, result, assumption, example, caveat, code, question. Each index entry is numbered `N.0.0` and each line `N.0.M`. The cutter checks those numbers against part order and line position, then stores the words after the number. A line may carry the same number in `data-n`. A code block must. The link is the heading. A missing or different `h2` is logged, and the link still wins. `data-src` on a line and `data-rel` on a section are stored with that part.

This page is shape 3. The hand-labeled set of about 100 replies is markdown only, scored with `structure_segment`. An HTML index reply is not a row in that file. A later set that holds both shapes labels each row with its shape version.
