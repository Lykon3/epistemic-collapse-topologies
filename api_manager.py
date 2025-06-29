import requests

class APIManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"

    def get_odds(self, sport_key="baseball_mlb", region="us", market="h2h"):
        url = f"{self.base_url}/sports/{sport_key}/odds/?apiKey={self.api_key}&regions={region}&markets={market}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code}, {response.text}")