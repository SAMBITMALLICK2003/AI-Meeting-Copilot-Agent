import os.path
import logging
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from ..config.settings import CALENDAR_SCOPES, CREDENTIALS_FILE, TOKEN_FILE

# Configure logging
logger = logging.getLogger(__name__)

def authenticate_google_calendar():
    """
    Handles the OAuth 2.0 authentication flow for Google Calendar API.

    Checks for existing valid tokens in token.json. If not found or invalid/expired,
    it initiates the OAuth flow using credentials_1.json. Saves the new tokens
    to token.json for future use.

    Returns:
        google.oauth2.credentials.Credentials: Validated credentials object.
        Returns None if authentication fails.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, CALENDAR_SCOPES)
            logger.info("Loaded existing credentials from token file")
        except Exception as e:
            logger.error(f"Error loading token file ({TOKEN_FILE}): {e}. Re-authenticating.")
            creds = None # Force re-authentication

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing access token...")
                creds.refresh(Request())
                logger.info("Access token refreshed successfully")
            except Exception as e:
                logger.error(f"Error refreshing token: {e}. Re-authenticating.")
                # Force re-authentication if refresh fails by deleting token file
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                creds = None # Ensure we proceed to the flow below

        # Only run the flow if creds are still None (initial run or refresh failed)
        if not creds:
             if not os.path.exists(CREDENTIALS_FILE):
                 logger.error(f"Error: Credentials file '{CREDENTIALS_FILE}' not found.")
                 return None
             try:
                 logger.info(f"Starting authentication flow using {CREDENTIALS_FILE}...")
                 flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, CALENDAR_SCOPES)
                 # port=0 allows it to find an available port automatically
                 creds = flow.run_local_server(port=0)
                 logger.info("Authentication successful.")
             except Exception as e:
                 logger.error(f"Error during authentication flow: {e}", exc_info=True)
                 return None

        # Save the credentials for the next run only if they are valid
        if creds and creds.valid:
            try:
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Credentials saved to {TOKEN_FILE}")
            except Exception as e:
                logger.error(f"Error saving token file: {e}", exc_info=True)

    if not creds or not creds.valid:
         logger.error("Failed to obtain valid credentials.")
         return None

    return creds
