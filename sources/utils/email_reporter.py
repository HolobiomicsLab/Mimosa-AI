"""
Email report sender for evaluation runs (Gmail API + OAuth2).

Sends a final-summary email through the Gmail API when configured.
Fully optional: silently no-ops if the Google client libs aren't installed
or if the OAuth credentials file isn't present, so it can be called
unconditionally at the end of an eval run.

Required:
    EMAIL_TO                 recipient(s), comma-separated
    credentials.json         OAuth client secrets, downloaded from Google Cloud Console
                             (path overridable via env var GMAIL_CREDENTIALS_FILE)
Optional env vars:
    EMAIL_USER               sender label (default 'me' = authenticated Gmail account)
    GMAIL_CREDENTIALS_FILE   path to OAuth client secrets (default 'credentials.json')
    GMAIL_TOKEN_FILE         path to cached user token (default 'token.json')

First run opens a browser for OAuth consent and caches the refresh token in
GMAIL_TOKEN_FILE so subsequent runs work headless. Run this module directly to
perform that one-time setup before launching long evaluations:

    python -m sources.utils.email_reporter
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# Google client libs are optional — import failures disable the reporter.
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    _GMAIL_AVAILABLE = True
    _IMPORT_ERROR: str | None = None
except ImportError as e:
    _GMAIL_AVAILABLE = False
    _IMPORT_ERROR = str(e)

_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Cache the Gmail service across calls so we only authenticate once per process.
_service = None
_service_init_attempted = False

import dotenv
dotenv.load_dotenv()


def _format_text_report(title: str, rows: list[tuple[str, str]]) -> str:
    key_w = max((len(k) for k, _ in rows), default=12) + 2
    lines = [title, "=" * max(len(title), 32), ""]
    for k, v in rows:
        if not v:
            lines.append("")
            lines.append(k)
        else:
            lines.append(f"{k.ljust(key_w)} {v}")
    return "\n".join(lines)


def _format_html_report(title: str, rows: list[tuple[str, str]]) -> str:
    body_rows = []
    for k, v in rows:
        if not v:
            body_rows.append(
                f'<tr><td colspan="2" style="padding-top:12px;'
                f'font-weight:bold;color:#666">{k}</td></tr>'
            )
        else:
            body_rows.append(
                f'<tr><td style="padding:4px 16px 4px 0;vertical-align:top">'
                f'<strong>{k}</strong></td>'
                f'<td style="padding:4px 0;font-family:monospace">{v}</td></tr>'
            )
    return (
        f'<html><body style="font-family:-apple-system,Segoe UI,sans-serif;font-size:14px">'
        f'<h2>{title}</h2>'
        f'<table style="border-collapse:collapse">{"".join(body_rows)}</table>'
        f'</body></html>'
    )


def _get_service():
    """Build (and cache) an authenticated Gmail API service, or return None.

    Returns None — and caches the failure so we don't retry every call — when
    Google client libs are missing, credentials.json is missing, or OAuth fails.
    """
    global _service, _service_init_attempted
    if _service is not None:
        return _service
    if _service_init_attempted:
        return None
    _service_init_attempted = True

    if not _GMAIL_AVAILABLE:
        logger.info(
            "[EMAIL] Skipping: Google client libs not installed (%s). "
            "Run: pip install google-auth google-auth-oauthlib google-api-python-client",
            _IMPORT_ERROR,
        )
        return None

    creds_path = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
    token_path = os.getenv("GMAIL_TOKEN_FILE", "token.json")

    if not os.path.exists(creds_path):
        logger.info(
            "[EMAIL] Skipping: OAuth credentials file '%s' not found "
            "(set GMAIL_CREDENTIALS_FILE or place credentials.json at the repo root).",
            creds_path,
        )
        return None

    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, _SCOPES)
        except Exception as e:
            logger.warning(f"[EMAIL] Could not load cached token at {token_path}: {e}")
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            logger.warning(f"[EMAIL] Token refresh failed, will re-auth: {e}")
            creds = None

    if not creds or not creds.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, _SCOPES)
            creds = flow.run_local_server(port=0)
        except Exception as e:
            logger.warning(f"[EMAIL] OAuth flow failed: {e}")
            return None
        try:
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            logger.info(f"[EMAIL] Cached refresh token to {token_path}")
        except Exception as e:
            logger.warning(f"[EMAIL] Could not save token to {token_path}: {e}")

    try:
        _service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        logger.warning(f"[EMAIL] Could not build Gmail service: {e}")
        return None
    return _service


def send_evaluation_report(
    subject: str,
    rows: list[tuple[str, str]],
    body_prefix: str = "",
) -> bool:
    """
    Send a summary email via the Gmail API.

    Args:
        subject: Email subject line.
        rows: List of (label, value) tuples. A row with an empty value is
              treated as a section header (matches print_summary semantics).
        body_prefix: Optional plain-text paragraph shown above the table.

    Returns:
        True if the email was sent, False otherwise (including the no-op case
        when libs or credentials are missing).
    """
    recipients_raw = os.getenv("EMAIL_TO")
    if not recipients_raw:
        logger.info("[EMAIL] Skipping report: EMAIL_TO not set.")
        return False

    service = _get_service()
    if service is None:
        return False

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    sender = os.getenv("EMAIL_USER", "me")  # 'me' = authenticated Gmail user

    text_body = _format_text_report(subject, rows)
    html_body = _format_html_report(subject, rows)
    if body_prefix:
        text_body = f"{body_prefix}\n\n{text_body}"
        html_body = f"<p>{body_prefix}</p>" + html_body

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        logger.info(f"[EMAIL] Report sent to {recipients}")
        return True
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send report: {e}")
        return False


if __name__ == "__main__":
    # One-time setup helper: triggers the OAuth browser flow and caches the
    # refresh token, then sends a test email if EMAIL_TO is configured.
    logging.basicConfig(level=logging.INFO)
    svc = _get_service()
    if svc is None:
        print("Gmail service not available — see logs above.")
        sys.exit(1)

    if not os.getenv("EMAIL_TO"):
        print("Auth OK and token cached. Set EMAIL_TO to send a test email.")
        sys.exit(0)

    ok = send_evaluation_report(
        subject="Mimosa email reporter — test",
        rows=[
            ("Steps evaluated", "3"),
            ("Successful runs", "2"),
            ("Success rate", "66.7%"),
            ("── ScienceAgentBench ──", ""),
            ("VER", "2/3 (66.7%)"),
            ("Total API Cost", "$0.1234"),
        ],
        body_prefix="This is a test email from sources/utils/email_reporter.py.",
    )
    print(f"sent={ok}")
