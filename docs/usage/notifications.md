# Notifications

Mimosa-AI can send phone notifications via [Pushover](https://pushover.net/)
on key events: each iteration completes, a new best workflow is found, or
the loop terminates. Optional — Mimosa runs fine without it.

## Setup

1. **Create a Pushover account** at [pushover.net](https://pushover.net/)
   and note your **User Key** from the dashboard.
2. **Create an application** named `Mimosa` — copy the **API Token**.
3. **Install the Pushover mobile app** on your phone and log in with the
   same account.
4. **Export the env vars** (or add them to `.env`):

   ```bash
   export PUSHOVER_TOKEN=your_api_token
   export PUSHOVER_USER=your_user_key
   ```

Mimosa picks them up automatically the next time you run it.

## What you'll get

- **Per-iteration completion** — score, iteration number, UUID.
- **New best** — every time the archive admits a new dominant workflow.
- **Final result** — the best UUID and its score at run termination.

Notifications are sent by [`sources/utils/notify.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/notify.py).
Errors talking to Pushover are logged and swallowed — they never crash a
run.

## Overriding from the CLI

If you want to use different credentials for a single run without editing
the env:

```bash
uv run main.py --task "…" \
  --pushover_token "$ALT_TOKEN" \
  --pushover_user "$ALT_USER"
```

## Disabling

Just unset `PUSHOVER_TOKEN` / `PUSHOVER_USER`. Mimosa silently skips
notification calls when either is missing.

## See also

- [API keys & environment](../getting-started/env.md) — full list of env vars.
- [Telemetry](telemetry.md) — span-level traces for in-IDE debugging.
