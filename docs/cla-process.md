# CLA Assistant Lite Process

## What this workflow does

- Runs CLA Assistant Lite on pull requests using `.github/workflows/cla.yml`.
- Uses `pull_request_target` to evaluate PRs in the base repository context.
- Prompts contributors in the PR when the short Individual Contributor Agreement is required.
- Records signatures in-repo after signing.

## CLA source and signature storage

- CLA document used by the action: `CLA/INDIVIDUAL_CLA.md` (via repository URL in the workflow).
- Signature storage path: `signatures/version1/cla.json`.
- Optional employer or institutional authorizations are handled manually by maintainers when needed.

## Required token/secret setup

- Required: `GITHUB_TOKEN` (provided automatically by GitHub Actions).
- No extra secret is needed for in-repo signature storage.
- Optional only for remote signature storage: `PERSONAL_ACCESS_TOKEN` with repo scope.
- Repository setting requirement: Actions `GITHUB_TOKEN` workflow permissions must allow read/write so the action can update signature records.

## Maintainer handling of failing CLA checks

- Do not merge PRs while the CLA check is failing.
- Ask contributors to complete the CLA prompt in the PR thread/checks.
- If a contribution is employment-linked or institution-owned, request `CLA/EMPLOYER_AUTHORIZATION.md` separately.
- Re-run the workflow by commenting `recheck` if needed.

## Branch protection recommendation

- Require the status check `CLA Check` before merge on protected branches.
- Keep this check required for external contribution branches.
