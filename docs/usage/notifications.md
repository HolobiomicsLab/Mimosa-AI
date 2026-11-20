# Notifications

Mimosa-AI can send phone notifications via [Pushover](https://pushover.net/)
on key events: each iteration completes, the loop terminates, or a workflow fails.
Optional — Mimosa runs fine without it.

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

- **Iteration completed** — iteration number, goal, cost, and per-run rewards.
- **Learning loop finished** — stop reason, final UUID, iteration count.
- **Task completed** — final UUID, when `--learn` is off.
- **Execution failed** — when a workflow fails to execute.
- **Planner notices** — sent from the planner's `--goal` flow.

Notifications are sent by [`sources/utils/notify.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/notify.py).
That module has no error handling of its own: a network failure while sending
is raised to the calling code.

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
