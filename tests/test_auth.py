# test_auth.py
import os
from auths import setup_google_auth
import config

print(f"Using project ID: {config.PROJECT_ID}")
print(f"Using credentials path: {config.DEFAULT_CREDENTIALS_PATH}")

credentials = setup_google_auth()
if credentials:
    print(f"Successfully authenticated using service account: {credentials.service_account_email}")
else:
    print("Using application default credentials")

print("Authentication test complete!")