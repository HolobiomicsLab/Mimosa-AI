// Shiki v4 fine-grained highlighter singleton — JavaScript regex engine (no
// wasm), one-dark-pro theme, only the languages the Observatory renders.
import { createHighlighterCore } from 'shiki/core'
import type { HighlighterCore } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'

export const THEME = 'one-dark-pro'

const LANGS = ['python', 'json', 'yaml', 'markdown', 'bash', 'r', 'diff'] as const
export type KnownLang = (typeof LANGS)[number]

const ALIASES: Record<string, KnownLang> = {
  py: 'python', python3: 'python',
  sh: 'bash', shell: 'bash', zsh: 'bash', shellscript: 'bash',
  yml: 'yaml',
  md: 'markdown',
  jsonc: 'json', json5: 'json',
  patch: 'diff',
}

/** Map a loose language name to a bundled grammar, or null when unknown. */
export function resolveLang(lang?: string | null): KnownLang | null {
  if (!lang) return null
  const l = lang.toLowerCase()
  if ((LANGS as readonly string[]).includes(l)) return l as KnownLang
  return ALIASES[l] ?? null
}

let instance: HighlighterCore | null = null
let promise: Promise<HighlighterCore> | null = null

function load(): Promise<HighlighterCore> {
  promise ??= createHighlighterCore({
    engine: createJavaScriptRegexEngine({ forgiving: true }),
    themes: [import('@shikijs/themes/one-dark-pro')],
    langs: [
      import('@shikijs/langs/python'),
      import('@shikijs/langs/json'),
      import('@shikijs/langs/yaml'),
      import('@shikijs/langs/markdown'),
      import('@shikijs/langs/bash'),
      import('@shikijs/langs/r'),
      import('@shikijs/langs/diff'),
    ],
    warnings: false,
  }).then((h) => {
    instance = h
    return h
  })
  return promise
}

/** True once the singleton highlighter has finished loading. */
export function isReady(): boolean {
  return instance !== null
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function plaintext(code: string): string {
  return `<pre class="shiki"><code>${escapeHtml(code)}</code></pre>`
}

/** Highlight `code`; unknown languages (or grammars the JS engine cannot
 * tokenize) fall back to escaped plaintext in the same HTML shape. */
export async function highlight(code: string, lang: string): Promise<string> {
  const resolved = resolveLang(lang)
  if (!resolved) return plaintext(code)
  const h = await load()
  try {
    return h.codeToHtml(code, { lang: resolved, theme: THEME })
  } catch {
    return plaintext(code)
  }
}
