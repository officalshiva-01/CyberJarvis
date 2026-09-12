import unittest
from unittest.mock import patch, MagicMock

from integrations.api_client import BaseApiClient
from integrations.github_api import GitHubClient
from integrations.weather_api import WeatherClient


class TestIntegrations(unittest.TestCase):
    @patch("httpx.Client")
    def test_api_client_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.example.com")
        res = client.get("/test")
        self.assertEqual(res, {"status": "ok"})

    @patch("httpx.Client")
    def test_api_client_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_response.text = "Not found"
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.example.com")
        res = client.get("/missing")
        self.assertIn("error", res)
        self.assertEqual(res["status_code"], 404)

    @patch.object(GitHubClient, "get")
    def test_github_client_repo(self, mock_get):
        mock_get.return_value = {
            "full_name": "shivaji/CyberJarvis",
            "description": "AI Assistant",
            "stargazers_count": 10,
            "forks_count": 2,
            "open_issues_count": 1,
            "default_branch": "main",
            "html_url": "https://github.com/shivaji/CyberJarvis",
        }
        gh = GitHubClient(token="mock_token")
        repo = gh.get_repo("shivaji", "CyberJarvis")
        self.assertEqual(repo["name"], "shivaji/CyberJarvis")
        self.assertEqual(repo["stars"], 10)

    @patch.object(WeatherClient, "get")
    def test_weather_client(self, mock_get):
        mock_get.return_value = {
            "current_condition": [{
                "temp_C": "24",
                "FeelsLikeC": "26",
                "weatherDesc": [{"value": "Partly cloudy"}],
                "humidity": "60",
                "windspeedKmph": "12",
            }],
            "nearest_area": [{
                "areaName": [{"value": "London"}],
                "country": [{"value": "United Kingdom"}],
            }]
        }
        wc = WeatherClient()
        weather = wc.get_weather("London")
        self.assertEqual(weather["location"], "London, United Kingdom")
        self.assertEqual(weather["temperature_c"], "24")
        self.assertEqual(weather["weather_desc"], "Partly cloudy")


if __name__ == "__main__":
    unittest.main()
