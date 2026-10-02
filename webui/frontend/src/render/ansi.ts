// ANSI escape sequences (agent observation logs are full of them).
// Covers CSI sequences (ESC [ ... cmd - colors, cursor moves) and lone ESC codes.
// eslint-disable-next-line no-control-regex -- matching ESC is the whole point
const ANSI_RE = /\u001b(?:\[[0-9;?]*[ -/]*[@-~]|[@-Z\\-_])/g

/** Remove ANSI escape sequences (ESC[...m colors and friends) from `text`. */
export function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, '')
}
