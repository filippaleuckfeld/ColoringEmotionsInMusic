import base64
import requests

class SpotifyClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    def get_access_token(self):
        """ Get Spotify access token """
        client_creds = f"{self.client_id}:{self.client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode()).decode()
        token_url = 'https://accounts.spotify.com/api/token'
        token_data = {"grant_type": "client_credentials"}
        token_headers = {"Authorization": f"Basic {client_creds_b64}"}

        try:
            response = requests.post(token_url, data=token_data, headers=token_headers)
            response.raise_for_status()
            token_response_data = response.json()
            self.access_token = token_response_data['access_token']
        except requests.exceptions.RequestException as e:
            print(f"Error getting access token: {e}")
            raise

    def get_audio_features(self, track_id):
        """ Get audio features for a track by its Spotify ID """
        if self.access_token is None:
            self.get_access_token()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        endpoint = f'https://api.spotify.com/v1/audio-features/{track_id}'

        try:
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting audio features: {e}")
            return None