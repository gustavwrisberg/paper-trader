import os
import base64
import smtplib
import requests
from email.mime.text import MIMEText


# Required secrets (set in GitHub repo → Settings → Secrets):
#
#   GMAIL_CLIENT_ID      — from Google Cloud Console OAuth2 credentials
#   GMAIL_CLIENT_SECRET  — same
#   GMAIL_REFRESH_TOKEN  — obtained once via oauth_setup.py (run locally)
#   EMAIL_FROM           — your Gmail address
#   EMAIL_TO             — destination address


def _get_access_token():
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id":     os.environ["GMAIL_CLIENT_ID"],
        "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
        "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
        "grant_type":    "refresh_token",
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]


def send_email(subject, body):
    sender = os.environ["EMAIL_FROM"]
    to     = os.environ["EMAIL_TO"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = to

    token = _get_access_token()

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        # XOAUTH2 expects a base64-encoded string: "user=<email>\x01auth=Bearer <token>\x01\x01"
        auth_str = f"user={sender}\x01auth=Bearer {token}\x01\x01"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        s.docmd("AUTH", f"XOAUTH2 {auth_b64}")
        s.send_message(msg)

    print(f"Email sent: {subject}")
