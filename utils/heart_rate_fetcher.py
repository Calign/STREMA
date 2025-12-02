# heart_rate_fetcher.py
import os
import csv
import datetime
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ['https://www.googleapis.com/auth/fitness.heart_rate.read']
CSV_FILE = 'heart_rate_data.csv'
DAYS_BACK = 7
HEART_RATE_CSV = "heart_rate_data.csv"

def get_credentials():
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def fetch_heart_rate_data():
    creds = get_credentials()
    service = build('fitness', 'v1', credentials=creds)

    end_time = int(datetime.datetime.now().timestamp() * 1e9)
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=DAYS_BACK)).timestamp() * 1e9)
    dataset = f"{start_time}-{end_time}"
    print(f"[fetch_heart_rate_data] Dataset range: {start_time}-{end_time}")

    data_sources = service.users().dataSources().list(userId='me').execute()
    heart_rate_source = None
    for ds in data_sources.get('dataSource', []):
        if 'heart_rate' in ds['dataType']['name']:
            heart_rate_source = ds['dataStreamId']
            break

    if not heart_rate_source:
        print("[fetch_heart_rate_data] ❌ No heart rate data source found.")
        return []

    print(f"[fetch_heart_rate_data] Using data source: {heart_rate_source}")

    data = service.users().dataSources().datasets().get(
        userId='me',
        dataSourceId=heart_rate_source,
        datasetId=dataset
    ).execute()

    points = []
    for point in data.get('point', []):
        hr_value = point['value'][0]['fpVal']
        ts = int(point['startTimeNanos']) / 1e9
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        points.append({'timestamp': time_str, 'heart_rate': hr_value})

    print(f"[fetch_heart_rate_data] Fetched {len(points)} points")
    return points


def save_to_csv(new_data):
    if not new_data:
        return
    new_df = pd.DataFrame(new_data)
    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    else:
        combined_df = new_df

    combined_df.sort_values(by='timestamp', inplace=True)
    combined_df.to_csv(CSV_FILE, index=False)
    print(f"✅ Heart rate CSV updated with latest data.")

LAST_HR_SYNC_OK = False

def update_heart_rate_csv():
    global LAST_HR_SYNC_OK
    try:
        new_data = fetch_heart_rate_data()
        if not new_data:
            LAST_HR_SYNC_OK = False
            return False

        # FIX: Save data to CSV
        save_to_csv(new_data)

        LAST_HR_SYNC_OK = True
        return True
    except Exception:
        LAST_HR_SYNC_OK = False
        print("[update_heart_rate_csv] Skipped: No connection or API error.")
        return False



def get_latest_hr_value_from_csv(csv_file=CSV_FILE):
    """Return most recent HR from CSV"""
    try:
        if not os.path.exists(csv_file):
            return None
        df = pd.read_csv(csv_file)
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest = df.sort_values('timestamp').iloc[-1]
        return float(latest['heart_rate'])
    except Exception as e:
        print(f"[HR CSV] Error reading {csv_file}: {e}")
        return None
