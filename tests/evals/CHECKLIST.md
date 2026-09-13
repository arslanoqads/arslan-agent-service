# How to create the golden dataset later

Do not fill this during the scaffold build. When you are ready:

1. Write public policy cases in `golden_set.public.json`.
   These cover routing and refusals only. No resume facts.
   Examples to write yourself: greeting calls no tool; email calls `send_resume_email`; calendar books one invite; job match must not invent a percentage; social links return only configured URLs; injection must not send mail or leak the system prompt.

2. Copy `golden_set.private.example.json` to `golden_set.private.json` (gitignored).
   Label a few resume questions against the local PDFs: which tool should run, and which phrase from a retrieved chunk must appear.

3. Mark seed rows `"source": "synthetic"`.

4. After the traces dashboard has real chats, add rows from those questions and mark them `"source": "production"`.

5. Run `python -m app.evals.runner`. An empty set exits cleanly. A case missing `input` fails the scaffold check.
