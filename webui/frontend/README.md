# Mimosa Observatory — frontend

Vite + React + TypeScript app for the Mimosa Observatory: the lineage tree,
replay viewer, workspace browser, and the setup/new-run pages. See
[`../README.md`](../README.md) for what the Observatory shows and the full
API surface; this file covers only running the frontend itself.

## Prerequisites

- Node >= 20

## Install

```bash
npm install
```

## Run (dev)

```bash
npm run dev
```

Serves at `http://localhost:5173`. The Vite dev server proxies `/api`
(REST endpoints and the `/api/live` WebSocket) to the backend on `:8848`
(see [`vite.config.ts`](./vite.config.ts)), so the app itself only ever uses
same-origin relative URLs — no CORS setup needed in dev.

Point the proxy at a different backend:

```bash
MIMOSA_API=http://host:port npm run dev
```

## Build

```bash
npm run build       # tsc -b && vite build -> static assets in dist/
npm run preview     # serve the build locally for a smoke check
```

## Lint

```bash
npm run lint        # oxlint
```

## Production notes

`npm run build` only produces static assets in `dist/`; nothing in this repo
serves them. The backend does not mount `dist/` as static files, and there is
no bundled reverse proxy, nginx config, or CI here. To deploy for real, serve
`dist/` from any static host or CDN and put it at the same origin as the
backend (via your own reverse proxy) — or point the built app at the
backend's URL directly. These pieces are not shipped; you provide them.

Keep in mind the Observatory is a single-operator, localhost tool with no
authentication — don't expose the backend (or a deployed frontend pointed at
it) on a shared or public network without adding your own access control.
