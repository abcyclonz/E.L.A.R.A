"""
One-time Google Calendar OAuth setup.

Run this ONCE on your local machine (not inside Docker) to generate token.json:

    cd agents/tools/assistant
    pip install google-auth-oauthlib google-api-python-client
    python auth_setup.py

It opens a browser for you to approve calendar access, then saves token.json
into google_auth/. The main server reads it from there at runtime.

Prerequisites:
  1. Go to console.cloud.google.com
  2. Create a project → Enable Google Calendar API
  3. OAuth consent screen → External → add your Gmail as test user
  4. Credentials → Create OAuth 2.0 Client ID → Desktop app → Download JSON
  5. Save the downloaded file as google_auth/credentials.json
  6. Run this script
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
AUTH_DIR = os.path.join(os.path.dirname(__file__), "google_auth")
CREDENTIALS_FILE = os.path.join(AUTH_DIR, "credentials.json")
TOKEN_FILE = os.path.join(AUTH_DIR, "token.json")


def main():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds.valid:
            print("Token is already valid — nothing to do.")
            return
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            print("Token refreshed and saved.")
            return

    if not os.path.exists(CREDENTIALS_FILE):
        print(f"ERROR: {CREDENTIALS_FILE} not found.")
        print("Download your OAuth 2.0 Client ID JSON from Google Cloud Console")
        print("and save it as google_auth/credentials.json")
        return

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\nDone! token.json saved to {TOKEN_FILE}")
    print("You can now docker compose up — the assistant tool will use Google Calendar.")


if __name__ == "__main__":
    main()
