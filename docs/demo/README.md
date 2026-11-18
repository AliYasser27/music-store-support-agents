# Demo recording

The README embeds `docs/demo/demo.gif`. Record it with the script below so a
reviewer can follow the system's behaviour without installing anything.

## Before recording

```bash
cp .env.example .env      # set GROQ_API_KEY
python -m src.run --demo
```

Run the demo once first so the model's latency is predictable on camera.

## Shot list

1. **Verified, combined request** — start a fresh session and send:
   *"My phone number is +55 (12) 3923-5555. How much was my most recent
   purchase? What albums do you have by the Rolling Stones?"*
   Point out the two routing log lines and the merged final answer.
2. **Memory recall** — same session:
   *"List some songs that match my preferences?"*
   Point out that the answer uses the preference saved in the previous turn.
3. **Unverified request** — start a new session and send the purchase question
   without credentials. Show the interrupt prompt, type `My customer id is 1`,
   and show the answer after the resume.
4. **Tests** — `pytest -m "not e2e"` passing.

## Encoding

Keep the GIF under ~10 MB for fast README loading:

```bash
ffmpeg -i recording.mp4 -vf "fps=10,scale=1100:-1:flags=lanczos" -loop 0 demo.gif
```
