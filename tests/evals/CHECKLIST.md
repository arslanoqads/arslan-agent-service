# How to create and extend the golden dataset

1. Write public policy cases in `golden_set.public.json`.
   These cover routing, refusals, and session follow-ups. No resume facts.
   Seeded production regressions include the email + meeting + "yes" dialogue.

2. Copy `golden_set.private.example.json` to `golden_set.private.json` (gitignored).
   Label a few resume questions against the local PDFs: which tool should run, and which phrase from a retrieved chunk must appear.

3. Mark seed rows `"source": "synthetic"`. Mark real chat failures `"source": "production"`.

4. Multi-turn cases may include `prior_turns` and `expected_route`.
   `python -m app.evals.runner` validates schema and deterministic route helpers.
   Full model scoring is still offline in this build.

5. After the traces dashboard has more chats, add rows from those questions.
