# What Graphirm is for

Graphirm runs a coding AI and keeps a checkable record of its work.

The model only reads and writes text. Graphirm is the program around it: it sends the model the task,
runs the commands the model asks for, sends back the results, and records all of it. Because every
reply and every action passes through Graphirm, Graphirm can record them in a form people can check.

## The problem

A chat with an AI is one long stream of text. Every sentence looks the same, whether it is a checked
fact, a plan, a guess, or a report of something that just happened. To trust the work you have to
reread it and judge each sentence yourself. To answer "why did this file change?" or "what was this
claim based on?" you have to dig through the whole conversation, and often the answer is not there.

## What Graphirm does about it

1. **Every reply is split into labeled pieces.** Each piece says what kind it is: a statement (true
   now), a result (done, and what happened), a recommendation, an assumption (believed, not checked),
   a question, a caveat, and so on. A reader can tell fact from plan from guess at a glance.
2. **Each piece is linked to its evidence.** A claim points to the file, command output or earlier
   piece it rests on. A claim with no link stands out.
3. **Every action is linked to the reply that justified it.** Commands, file changes and approvals are
   recorded and tied to the piece that asked for them, and to who approved them.

Together this makes a chain you can follow in both directions: from a change, to the recommendation
behind it, to the result it relied on, to the test output that showed it.

## Who it is for

- **The person working with the AI:** know what to trust and what to check, without rereading
  everything.
- **The AI itself, later:** pull back "the decision we made about X" by lookup, instead of carrying the
  whole history in every call.
- **Other agents:** hand over exactly the pieces another agent needs.
- **Auditors and regulated teams:** answer what was done, why, based on what, and who approved it,
  from the record.

## What it is not

- It is not a claim to be cheaper or smarter on small, single bug fixes. Measured on those, a plain
  agent with no memory does as well for less (`docs/journal.md`, 2026-11-07).
- It is not "memory by packing the past into every call". That was measured: same results, about 6×
  the cost. Earlier work is reached by lookup.

The benefit is expected on large, long-running work and where the work has to be accounted for. It is
judged there, as cost per solved task and cost per answered audit question.

## How we prove it

In order, each step only after the one before it passes
(`docs/plans/2026-11-07-prove-the-structure.md`):

0. **The structure works on its own terms:** pieces labeled right, links real and pointing at the
   right evidence, earlier pieces readable and findable, nothing missing from the record.
1. **It helps the AI:** it beats the same agent keeping a notes file.
2. **It pays off on large work:** it beats a plain agent as the work outgrows the window.
3. **It makes the AI accountable:** an auditor answers the standard questions from the record.

## Where it stands (2026-11-08)

- Result: step 0.4, the completeness check, fails (`docs/plans/2026-11-07-structure-completeness.md`).
  Tool results and `write` / `edit` changes are recorded completely.
- Result: approvals are not recorded in headless runs, and an approval does not say whether a human or
  auto-approve gave it.
- Result: most file changes go through shell commands, and those changes are not recorded.
- Result: replies keep their pieces in new sessions (53 of 56) but mostly lose them in long sessions,
  and text written next to tool calls is never split.
- Statement: links to evidence are stored as metadata, not as graph edges, and whether they point at
  the right evidence is not checked yet.
- Recommendation: fix these first (`docs/backlog.md`, under "Prove the structure"), then rerun step 0
  on a `main` binary before any comparison with a plain agent.
