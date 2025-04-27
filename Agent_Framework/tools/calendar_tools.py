import datetime
import logging
import pytz
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from ..utils.google_auth import authenticate_google_calendar
from ..tools.base_tool import BaseTool
from ..utils.microsoft_auth import authenticate_microsoft


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GoogleCalendarTool(BaseTool):
    """Base class for Google Calendar tools"""

    def __init__(self, name, description):
        super().__init__(name, description)
        self.service = None

    async def initialize(self):
        """Initialize Google Calendar API connection"""
        try:
            creds = authenticate_google_calendar()
            if not creds:
                logger.error("Google Calendar authentication failed")
                return False

            self.service = build('calendar', 'v3', credentials=creds)
            logger.info(f"Successfully initialized {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar tool: {e}")
            return False


class ScheduleGoogleMeetingTool(GoogleCalendarTool):
    """Tool for scheduling Google Meet meetings"""

    def __init__(self):
        super().__init__(
            name="ScheduleMeeting",
            description="Schedules a Google Meet meeting and returns the meeting link"
        )

    async def execute(self, date_str, time_str, summary="Meeting via Script",
                     description="", duration_minutes=60, attendees=None, timezone='UTC'):
        """
        Schedules a meeting with a Google Meet link in Google Calendar.

        Args:
            date_str (str): The date in 'YYYY-MM-DD' format.
            time_str (str): The time in 'HH:MM:SS' format (24-hour).
            summary (str): The title/summary of the calendar event.
            description (str): The description for the event.
            duration_minutes (int): Duration of the meeting in minutes.
            attendees (list[str], optional): List of email addresses for attendees. Defaults to None.
            timezone (str, optional): The timezone for the start/end times.

        Returns:
            dict: The created event object from the Google Calendar API, including the Meet link.
                  Returns None if scheduling fails.
        """
        if not self.service:
            if not await self.initialize():
                return {
                    "status": "error",
                    "message": "Failed to initialize Google Calendar service"
                }

        try:
            # Combine date and time, parse into datetime object
            start_datetime_naive = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")

            # Create end time by adding duration
            start_datetime = start_datetime_naive
            end_datetime = start_datetime + datetime.timedelta(minutes=duration_minutes)

            # Format for the API
            start_time_formatted = start_datetime.isoformat()
            end_time_formatted = end_datetime.isoformat()

            event_body = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time_formatted,
                    'timeZone': timezone,
                },
                'end': {
                    'dateTime': end_time_formatted,
                    'timeZone': timezone,
                },
                # Add attendees if provided
                'attendees': [{'email': email} for email in attendees] if attendees else [],
                # Request a Google Meet conference link
                'conferenceData': {
                    'createRequest': {
                        # Using a simple unique ID based on time
                        'requestId': f"meet-req-{datetime.datetime.now().timestamp()}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'reminders': {
                    'useDefault': True, # Use calendar's default reminders
                },
            }

            logger.info(f"Creating event: '{summary}' on {date_str} at {time_str} ({timezone})")

            # Call the Calendar API
            # conferenceDataVersion=1 ensures Meet link generation
            created_event = self.service.events().insert(
                calendarId='primary', # Use 'primary' for the user's main calendar
                body=event_body,
                conferenceDataVersion=1
            ).execute()

            meet_link = created_event.get('hangoutLink', 'No Meet link generated.')
            event_link = created_event.get('htmlLink', 'No event link available.')
            success_msg = f"Event '{summary}' created successfully."
            logger.info(f"{success_msg} Event Link: {event_link}, Meet Link: {meet_link}")

            # Return structured response
            return {
                "status": "success",
                "message": success_msg,
                "summary": summary,
                "date": date_str,
                "time": time_str,
                "timezone": timezone,
                "event_link": event_link,
                "meet_link": meet_link
            }

        except HttpError as error:
            logger.error(f"API error occurred: {error}")
            logger.error(f"Details: {error.content}")
            return {
                "status": "error",
                "message": f"API error: {error}",
                "details": str(error.content)
            }
        except ValueError as ve:
            logger.error(f"Error parsing date/time: {ve}")
            return {
                "status": "error",
                "message": f"Error parsing date/time: {ve}. Format should be YYYY-MM-DD and HH:MM:SS."
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {e}"
            }


class GetGoogleMeetingsTool(GoogleCalendarTool):
    """Tool for retrieving scheduled meetings from Google Calendar"""

    def __init__(self):
        super().__init__(
            name="GetMeetings",
            description="Retrieves scheduled meetings from Google Calendar"
        )

    async def execute(self, start_date_str, start_time_str, end_date_str, end_time_str,
                     timezone='UTC', calendar_id='primary', max_results=50):
        """
        Retrieves scheduled meetings from Google Calendar within a specified time range.

        Args:
            start_date_str (str): The start date in 'YYYY-MM-DD' format.
            start_time_str (str): The start time in 'HH:MM:SS' format (24-hour).
            end_date_str (str): The end date in 'YYYY-MM-DD' format.
            end_time_str (str): The end time in 'HH:MM:SS' format (24-hour).
            timezone (str, optional): The timezone for interpreting the start/end times.
            calendar_id (str, optional): The ID of the calendar to query. Defaults to 'primary'.
            max_results (int, optional): Maximum number of events to retrieve. Defaults to 50.

        Returns:
            dict: A dictionary containing the status and a list of events found.
        """
        if not self.service:
            if not await self.initialize():
                return {
                    "status": "error",
                    "message": "Failed to initialize Google Calendar service"
                }

        try:
            # Handle timezone
            try:
                tz = pytz.timezone(timezone)
            except pytz.UnknownTimeZoneError:
                errmsg = f"Invalid timezone specified: '{timezone}'. Please use a valid IANA timezone name."
                logger.error(errmsg)
                return {"status": "error", "message": errmsg}

            try:
                # Combine date and time, create naive datetime objects
                start_dt_naive = datetime.datetime.fromisoformat(f"{start_date_str}T{start_time_str}")
                end_dt_naive = datetime.datetime.fromisoformat(f"{end_date_str}T{end_time_str}")

                # Make datetime objects timezone-aware using the specified timezone
                start_dt_aware = tz.localize(start_dt_naive)
                end_dt_aware = tz.localize(end_dt_naive)

                # Format for the Google Calendar API (RFC3339 format)
                time_min = start_dt_aware.isoformat()
                time_max = end_dt_aware.isoformat()

            except ValueError as ve:
                errmsg = f"Error parsing date/time: {ve}. Ensure format is YYYY-MM-DD and HH:MM:SS."
                logger.error(errmsg)
                return {"status": "error", "message": errmsg}

            logger.info(f"Retrieving events from {time_min} to {time_max} in calendar '{calendar_id}'")

            # Call the Calendar API's events.list method
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,        # Expands recurring events into instances
                orderBy='startTime'       # Orders events by their start time
            ).execute()

            events = events_result.get('items', [])
            retrieved_events = []

            if not events:
                logger.info("No upcoming events found in the specified range.")
                return {"status": "success", "count": 0, "events": []}

            logger.info(f"Found {len(events)} events.")
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date')) # Handles all-day events
                end = event['end'].get('dateTime', event['end'].get('date'))
                attendees = [att.get('email') for att in event.get('attendees', []) if att.get('email')]

                event_data = {
                    'summary': event.get('summary', 'No Title'),
                    'start_time': start,
                    'end_time': end,
                    'meet_link': event.get('hangoutLink', None),
                    'event_link': event.get('htmlLink', None),
                    'description': event.get('description', None),
                    'attendees': attendees,
                    'id': event.get('id')
                }
                retrieved_events.append(event_data)

            return {
                "status": "success",
                "count": len(retrieved_events),
                "events": retrieved_events
            }

        except HttpError as error:
            errmsg = f"An API error occurred: {error}"
            logger.error(f"{errmsg} - Details: {error.content}")
            return {"status": "error", "message": errmsg, "details": str(error.content)}
        except Exception as e:
            errmsg = f"An unexpected error occurred: {e}"
            logger.error(errmsg, exc_info=True)
            return {"status": "error", "message": errmsg}


class MicrosoftGraphTool(BaseTool):
    def __init__(self, name, description):
        super().__init__(name, description)
        self.access_token = None

    def authenticate_microsoft_graph(self):
        """Use microsoft_auth to authenticate."""
        creds = authenticate_microsoft()  # returns a dictionary with credentials
        if 'access_token' not in creds:
            logger.error("Microsoft authentication failed.")
            return False

        self.access_token = creds['access_token']  # Save the access token to the instance
        return True

    async def initialize(self):
        """Initialize Microsoft Graph API connection"""
        if not self.authenticate_microsoft_graph():
            logger.error("Microsoft authentication failed.")
            return False

        logger.info(f"Successfully initialized {self.name}")
        return True


class GetTeamsMeetingsTool(MicrosoftGraphTool):
    """Tool for retrieving scheduled meetings from Microsoft Teams Calendar"""

    def __init__(self):
        super().__init__(
            name="GetMeetingsTeams",
            description="Retrieves scheduled meetings from Microsoft Teams Calendar using Microsoft Graph API"
        )

    async def execute(self, start_date_str, start_time_str, end_date_str, end_time_str,
                      timezone='UTC', calendar_id='primary', max_results=50):
        """
        Retrieves scheduled meetings from Microsoft Teams Calendar within a specified time range.

        Args:
            start_date_str (str): The start date in 'YYYY-MM-DD' format.
            start_time_str (str): The start time in 'HH:MM:SS' format (24-hour).
            end_date_str (str): The end date in 'YYYY-MM-DD' format.
            end_time_str (str): The end time in 'HH:MM:SS' format (24-hour).
            timezone (str, optional): The timezone for interpreting the start/end times.
            calendar_id (str, optional): Ignored for Microsoft (kept for compatibility).
            max_results (int, optional): Maximum number of events to retrieve. Defaults to 50.

        Returns:
            dict: A dictionary containing the status and a list of events found.
        """
        if not await self.initialize():
            return {
                "status": "error",
                "message": "Failed to authenticate with Microsoft Graph"
            }

        try:
            # Handle timezone
            try:
                tz = pytz.timezone(timezone)
            except pytz.UnknownTimeZoneError:
                errmsg = f"Invalid timezone specified: '{timezone}'. Please use a valid IANA timezone name."
                logger.error(errmsg)
                return {"status": "error", "message": errmsg}

            try:
                # Combine date and time, create naive datetime objects
                start_dt_naive = datetime.datetime.fromisoformat(f"{start_date_str}T{start_time_str}")
                end_dt_naive = datetime.datetime.fromisoformat(f"{end_date_str}T{end_time_str}")

                # Make datetime objects timezone-aware using the specified timezone
                start_dt_aware = tz.localize(start_dt_naive)
                end_dt_aware = tz.localize(end_dt_naive)

                # Format in UTC for Microsoft Graph (Graph expects UTC time for filters)
                time_min = start_dt_aware.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                time_max = end_dt_aware.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

            except ValueError as ve:
                errmsg = f"Error parsing date/time: {ve}. Ensure format is YYYY-MM-DD and HH:MM:SS."
                logger.error(errmsg)
                return {"status": "error", "message": errmsg}

            logger.info(f"Retrieving events from {time_min} to {time_max} from Microsoft Teams calendar")

            params = {
                '$select': 'subject,start,end,isOnlineMeeting,onlineMeeting,location,attendees',
                '$filter': f"start/dateTime ge '{time_min}' and start/dateTime le '{time_max}'",
                '$orderby': 'start/dateTime',
                '$top': str(max_results)
            }

            response = requests.get(
                'https://graph.microsoft.com/v1.0/me/events',
                headers={'Authorization': f'Bearer {self.access_token}'},
                params=params
            )
            response.raise_for_status()
            events = response.json().get('value', [])
            retrieved_events = []

            if not events:
                logger.info("No upcoming events found in the specified range.")
                return {"status": "success", "count": 0, "events": []}

            logger.info(f"Found {len(events)} events.")
            for event in events:
                start = event.get('start', {}).get('dateTime')
                end = event.get('end', {}).get('dateTime')
                attendees = [att.get('emailAddress', {}).get('address') for att in event.get('attendees', []) if att.get('emailAddress')]
                teams_link = None
                if event.get('onlineMeeting') is not None:
                    teams_link = event['onlineMeeting'].get('joinUrl', 'No Teams link generated.')
                else:
                    teams_link = 'No Teams link available.'

                event_data = {
                    'summary': event.get('subject', 'No Title'),
                    'start_time': start,
                    'end_time': end,
                    'meet_link': teams_link if event.get('isOnlineMeeting') else 'No Teams link available.',
                    'event_link': None,
                    'description': event.get('bodyPreview'),
                    'attendees': attendees,
                    'id': event.get('id')
                }
                retrieved_events.append(event_data)

            return {
                "status": "success",
                "count": len(retrieved_events),
                "events": retrieved_events
            }

        except requests.exceptions.HTTPError as error:
            errmsg = f"An API error occurred: {error}"
            logger.error(f"{errmsg} - Details: {error.response.text}")
            return {"status": "error", "message": errmsg, "details": str(error.response.text)}
        except Exception as e:
            errmsg = f"An unexpected error occurred: {e}"
            logger.error(errmsg, exc_info=True)
            return {"status": "error", "message": errmsg}


class ScheduleTeamsMeetingTool(MicrosoftGraphTool):
    """Tool for scheduling Microsoft Teams meetings"""

    def __init__(self):
        super().__init__(
            name="ScheduleTeamsMeeting",
            description="Schedules a Microsoft Teams meeting and returns the meeting link"
        )

    async def execute(self, date_str, time_str, summary="Meeting via Script", description="",
                      duration_minutes=60, attendees=None, timezone='UTC'):
        """
        Schedules a Microsoft Teams meeting in the Calendar.

        Args:
            date_str (str): The date in 'YYYY-MM-DD' format.
            time_str (str): The time in 'HH:MM:SS' format (24-hour).
            summary (str): The title/summary of the calendar event.
            description (str): The description for the event.
            duration_minutes (int): Duration of the meeting in minutes.
            attendees (list[str], optional): List of email addresses for attendees. Defaults to None.
            timezone (str, optional): The timezone for the start/end times.

        Returns:
            dict: The created event object from the Microsoft Graph API, including the Teams meeting link.
                  Returns an error dict if scheduling fails.
        """
        if not await self.initialize():
            return {
                "status": "error",
                "message": "Failed to authenticate with Microsoft Graph"
            }

        try:
            # Combine date and time into datetime
            start_datetime_naive = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")
            start_datetime = start_datetime_naive
            end_datetime = start_datetime + datetime.timedelta(minutes=duration_minutes)

            # Format as ISO strings
            start_time_formatted = start_datetime.isoformat()
            end_time_formatted = end_datetime.isoformat()

            # Prepare attendees if provided
            attendees_list = [{'emailAddress': {'address': email}} for email in attendees] if attendees else []

            event_body = {
                "subject": summary,
                "body": {
                    "contentType": "HTML",
                    "content": description
                },
                "start": {
                    "dateTime": start_time_formatted,
                    "timeZone": timezone
                },
                "end": {
                    "dateTime": end_time_formatted,
                    "timeZone": timezone
                },
                "attendees": attendees_list,
                "location": {
                    "displayName": "Microsoft Teams Meeting"
                },
                "isOnlineMeeting": True,
                "onlineMeetingProvider": "teamsForBusiness"
            }

            # Log event creation attempt
            logger.info(f"Creating event: '{summary}' on {date_str} at {time_str} ({timezone})")

            # Send POST request to Microsoft Graph API to create the event
            created_event = requests.post(
                'https://graph.microsoft.com/v1.0/me/events',
                headers={
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                },
                json=event_body
            )

            if created_event.status_code == 201:
                event_data = created_event.json()
                teams_link = event_data['onlineMeeting']['joinUrl'] if event_data['isOnlineMeeting'] else None
                event_link = event_data.get('webLink', 'No event link available.')

                success_msg = f"Event '{summary}' created successfully."
                logger.info(f"{success_msg} Event Link: {event_link}, Teams Link: {teams_link}")

                return {
                    "status": "success",
                    "message": success_msg,
                    "summary": summary,
                    "date": date_str,
                    "time": time_str,
                    "timezone": timezone,
                    "event_link": event_link,
                    "teams_link": teams_link
                }
            else:
                logger.error(f"Failed to create event. Status Code: {created_event.status_code}")
                return {
                    "status": "error",
                    "message": f"Failed to create event. Status Code: {created_event.status_code}"
                }

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {e}"
            }


class GetMeetingsTool(MicrosoftGraphTool, GoogleCalendarTool):
    """Tool for retrieving scheduled meetings from either Microsoft Teams or Google Calendar"""

    def __init__(self):
        super().__init__(
            name="GetMeetings",
            description="Retrieves scheduled meetings from Microsoft Teams or Google Calendar based on the platform."
        )

    async def execute(self, platform, start_date_str, start_time_str, end_date_str, end_time_str,
                      timezone='UTC', calendar_id='primary', max_results=50):
        """
        Retrieves scheduled meetings from either Microsoft Teams or Google Calendar based on the platform parameter.

        Args:
            platform (str): The platform to use, either 'microsoft' or 'google'.
            start_date_str (str): The start date in 'YYYY-MM-DD' format.
            start_time_str (str): The start time in 'HH:MM:SS' format (24-hour).
            end_date_str (str): The end date in 'YYYY-MM-DD' format.
            end_time_str (str): The end time in 'HH:MM:SS' format (24-hour).
            timezone (str, optional): The timezone for interpreting the start/end times.
            calendar_id (str, optional): Calendar ID to use, 'primary' by default for Google Calendar.
            max_results (int, optional): Maximum number of events to retrieve. Defaults to 50.

        Returns:
            dict: A dictionary containing the status and a list of events found.
        """
        if platform.lower() == 'microsoft':
            # Use Microsoft Graph API to retrieve meetings
            ms_tool = GetTeamsMeetingsTool()
            return await ms_tool.execute(start_date_str, start_time_str, end_date_str, end_time_str,
                                         timezone, calendar_id, max_results)

        elif platform.lower() == 'google':
            # Use Google Calendar API to retrieve meetings
            google_tool = GetGoogleMeetingsTool()
            return await google_tool.execute(start_date_str, start_time_str, end_date_str, end_time_str,
                                             timezone, calendar_id, max_results)

        else:
            return {
                "status": "error",
                "message": "Invalid platform specified. Please choose 'microsoft' or 'google'."
            }


class ScheduleMeetingTool(MicrosoftGraphTool, GoogleCalendarTool):
    """Tool for scheduling meetings using either Google Calendar or Microsoft Teams."""

    def __init__(self):
        super().__init__(
            name="ScheduleMeeting",
            description="Schedules a meeting using either Google Calendar or Microsoft Teams"
        )
        self.microsoft_tool = ScheduleTeamsMeetingTool()
        self.google_tool = ScheduleGoogleMeetingTool()

    async def execute(self, platform, date_str, time_str, summary="Meeting", description="",
                      duration_minutes=60, attendees=None, timezone='UTC'):
        """
        Schedules a meeting based on the specified platform.

        Args:
            platform (str): The platform to use for scheduling the meeting ('google' or 'microsoft').
            date_str (str): The date in 'YYYY-MM-DD' format.
            time_str (str): The time in 'HH:MM:SS' format (24-hour).
            summary (str): The title/summary of the calendar event.
            description (str): The description for the event.
            duration_minutes (int): Duration of the meeting in minutes.
            attendees (list[str], optional): List of email addresses for attendees.
            timezone (str, optional): The timezone for the start/end times.

        Returns:
            dict: The created event object from the selected platform, including the meeting link.
        """
        if platform.lower() == "microsoft":
            # Use Microsoft tool
            return await self.microsoft_tool.execute(
                date_str, time_str, summary, description, duration_minutes, attendees, timezone
            )
        elif platform.lower() == "google":
            # Use Google tool
            return await self.google_tool.execute(
                date_str, time_str, summary, description, duration_minutes, attendees, timezone
            )
        else:
            return {
                "status": "error",
                "message": f"Invalid platform '{platform}' specified. Use 'google' or 'microsoft'."
            }