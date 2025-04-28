import os.path
import logging
import requests
import time
import json

from Agent_Framework.config.settings import MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID, MS_REDIRECT_URI, MS_SCOPES, MS_TOKEN_FILE

# Microsoft Identity Platform endpoints
AUTHORITY = 'https://login.microsoftonline.com/common'
AUTH_URL = f"{AUTHORITY}/oauth2/v2.0/authorize"
TOKEN_URL = f"{AUTHORITY}/oauth2/v2.0/token"

# Configure logging
logger = logging.getLogger(__name__)


def authenticate_microsoft():
    """
    Handles the OAuth 2.0 authentication flow for Microsoft Graph API.

    - Checks for existing valid tokens in token.json. If not found or expired,
      it will initiate the OAuth flow using the client credentials.
    - Saves the new tokens to token.json for future use.

    Returns:
        dict: Validated access token.
        None: If authentication fails.
    """
    creds = None

    # Check if the token file exists
    if os.path.exists(MS_TOKEN_FILE):
        try:
            with open(MS_TOKEN_FILE, 'r') as token_file:
                creds = json.load(token_file)  # Deserialize the token data
            # If the token has expired, refresh it
            if creds['expires_at'] < time.time():
                logger.info("Access token expired. Attempting to refresh...")
                creds = refresh_token(creds)  # Try to refresh the token
        except Exception as e:
            logger.error(f"Error loading token file ({MS_TOKEN_FILE}): {e}. Re-authenticating.")
            creds = None

    # If no valid credentials are available, initiate OAuth flow
    if not creds:
        creds = oauth_flow()

    # If successful, save the new token
    if creds:
        # Calculate expiration time based on expires_in (in seconds)
        expires_in = creds['expires_in']
        creds['expires_at'] = time.time() + expires_in  # Set expiry time for the token

        try:
            with open(MS_TOKEN_FILE, 'w') as token_file:
                json.dump(creds, token_file)  # Save new token data using JSON
            logger.info(f"Credentials saved to {MS_TOKEN_FILE}")
        except Exception as e:
            logger.error(f"Error saving token file: {e}")

    if not creds:
        logger.error("Authentication failed.")

    return creds


def refresh_token(creds):
    """
    Refresh the Microsoft access token using the refresh token.

    Args:
        creds (dict): The credentials containing the refresh token.

    Returns:
        dict: New credentials with an updated access token.
        None: If refreshing the access token fails.
    """
    try:
        # Prepare the data to refresh the token using the refresh token
        token_data = {
            'client_id': MS_CLIENT_ID,
            'client_secret': MS_CLIENT_SECRET,
            'refresh_token': creds['refresh_token'],
            'grant_type': 'refresh_token',
            'scope': ' '.join(MS_SCOPES),
        }

        # Make a POST request to refresh the access token
        response = requests.post(TOKEN_URL, data=token_data)

        # Check if the response is successful
        if response.status_code != 200:
            logger.error(f"Token refresh failed: {response.text}")
            return None

        # If successful, extract the new access token and other details
        new_creds = response.json()

        if 'access_token' in new_creds:
            # Update the credentials with the new access token and expiry time
            creds['access_token'] = new_creds['access_token']
            creds['expires_at'] = time.time() + new_creds.get('expires_in', 3600)  # Set expiration time
            logger.info("Access token refreshed successfully.")
            return creds
        else:
            logger.error(f"Error refreshing access token: {new_creds.get('error_description', 'Unknown error')}")
            return None

    except Exception as e:
        logger.error(f"Error refreshing token: {e}", exc_info=True)
        return None


##################################################

import requests
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

auth_code_global = None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code_global
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)

        if 'code' in params:
            auth_code_global = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<h1>Authentication successful. You can close this window.</h1>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed. No code found.</h1>")

    def log_message(self, format, *args):
        # Suppress HTTP server logs
        return


def run_local_server(server_class=HTTPServer, handler_class=OAuthCallbackHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.handle_request()  # Only handle a single request, then exit


def oauth_flow():
    """
    Initiates the Microsoft OAuth flow and obtains an access token automatically.
    """
    try:
        # Prepare authentication URL
        auth_params = {
            'client_id': MS_CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': MS_REDIRECT_URI,
            'scope': ' '.join(MS_SCOPES)
        }
        auth_request = requests.Request('GET', AUTH_URL, params=auth_params).prepare()

        # Start local server to catch the redirect
        server_thread = threading.Thread(target=run_local_server)
        server_thread.start()

        # Open the URL automatically in default browser
        print(f"Opening browser for authentication...")
        webbrowser.open(auth_request.url)

        # Wait for the server to get the auth code
        server_thread.join()

        if not auth_code_global:
            logger.error("Authorization failed: No code returned")
            return None

        # Prepare data to exchange authorization code for token
        token_data = {
            'client_id': MS_CLIENT_ID,
            'client_secret': MS_CLIENT_SECRET,
            'code': auth_code_global,
            'redirect_uri': MS_REDIRECT_URI,
            'grant_type': 'authorization_code',
            'scope': ' '.join(MS_SCOPES)
        }

        # Exchange code for access token
        response = requests.post(TOKEN_URL, data=token_data)

        if response.status_code != 200:
            logger.error(f"Token request failed: {response.text}")
            return None

        creds = response.json()
        if 'access_token' in creds:
            logger.info("Authentication successful.")
            return creds
        else:
            logger.error(f"Error obtaining access token: {creds.get('error_description', 'Unknown error')}")
            return None

    except Exception as e:
        logger.error(f"Error during authentication flow: {e}", exc_info=True)
        return None


##################################################

# def oauth_flow():
#     """
#     Initiates the Microsoft OAuth flow and obtains an access token.

#     Returns:
#         dict: The newly obtained access token along with refresh token and expiration time.
#         None: If authentication fails.
#     """
#     try:
#         # Prepare authentication URL with required parameters
#         auth_params = {
#             'client_id': MS_CLIENT_ID,
#             'response_type': 'code',
#             'redirect_uri': MS_REDIRECT_URI,
#             'scope': ' '.join(MS_SCOPES)
#         }
#         auth_request = requests.Request('GET', AUTH_URL, params=auth_params).prepare()

#         # Direct the user to the authentication page (open in a browser)
#         print(f"Visit this URL to authenticate: {auth_request.url}")

#         # User authenticates and provides authorization code (manual step)
#         auth_code = input("Enter the authorization code: ")

#         if not auth_code:
#             logger.error("Authorization failed: No code returned")
#             return None

#         # Prepare data to exchange authorization code for an access token
#         token_data = {
#             'client_id': MS_CLIENT_ID,
#             'client_secret': MS_CLIENT_SECRET,
#             'code': auth_code,
#             'redirect_uri': MS_REDIRECT_URI,
#             'grant_type': 'authorization_code',
#             'scope': ' '.join(MS_SCOPES)
#         }

#         # Make a POST request to get the access token
#         response = requests.post(TOKEN_URL, data=token_data)

#         if response.status_code != 200:
#             logger.error(f"Token request failed: {response.text}")
#             return None

#         # If the token request is successful, extract the access token
#         creds = response.json()
#         if 'access_token' in creds:
#             logger.info("Authentication successful.")
#             return creds
#         else:
#             logger.error(f"Error obtaining access token: {creds.get('error_description', 'Unknown error')}")
#             return None

#     except Exception as e:
#         logger.error(f"Error during authentication flow: {e}", exc_info=True)
#         return None

def get_upcoming_meetings(creds):
    """
    Fetches upcoming meetings from the user's Microsoft calendar using Graph API.

    Args:
        creds (dict): Validated credentials containing the access token.

    Returns:
        list: A list of upcoming meetings/events.
    """
    access_token = creds['access_token']
    url = "https://graph.microsoft.com/v1.0/me/events"

    # Set up the headers with the access token
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Send the request to the Microsoft Graph API to fetch events
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        events = response.json()  # Parse the response JSON
        logger.info("Fetched events successfully.")

        if 'value' in events:
            upcoming_meetings = events['value']
            return upcoming_meetings
        else:
            logger.warning("No events found.")
            return []

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        return []
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
        return []


def main():
    # Authenticate Microsoft and retrieve the access token
    creds = authenticate_microsoft()

    # If authentication is successful, retrieve meetings
    if creds:
        meetings = get_upcoming_meetings(creds)

        if meetings:
            print("Upcoming Meetings:")
            for meeting in meetings:
                print(f"Subject: {meeting['subject']}")
                print(f"Start: {meeting['start']['dateTime']}")
                print(f"End: {meeting['end']['dateTime']}")
                print("-" * 50)
        else:
            print("No upcoming meetings found.")
    else:
        print("Failed to authenticate with Microsoft.")


if __name__ == "__main__":
    main()
