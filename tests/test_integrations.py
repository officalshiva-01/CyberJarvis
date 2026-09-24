import unittest
from unittest.mock import patch, MagicMock
import httpx

from integrations.api_client import BaseApiClient, redact_secrets
from integrations.github_api import (
    GitHubClient,
    get_repository,
    get_repository_status,
    get_recent_commits,
    DEFAULT_OWNER,
    DEFAULT_REPO,
)
from integrations.weather_api import WeatherClient, get_current_weather
from agent.router import ExecutionRouter, RouteType
from agent.task_profile import classify_agent_task_profile, AgentTaskProfile
from agent.agent_core import AgentCore
from tools.registry import registry
from security.audit import audit_logger


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
        self.assertEqual(res["status"], "http_error")

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
        self.assertTrue(repo["success"])

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
        self.assertTrue(weather["success"])

    # 1. API client timeout test
    @patch("httpx.Client")
    def test_api_client_timeout(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ReadTimeout("Read timed out")
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.github.com", timeout=2.0, max_retries=0)
        res = client.get("/repos/test/repo")
        self.assertIn("error", res)
        self.assertEqual(res.get("status"), "timeout")
        self.assertTrue(res.get("timed_out"))
        self.assertFalse(res.get("success", False))

    # 2. Missing credentials & auth error test
    @patch("httpx.Client")
    def test_missing_credentials_and_auth_error(self, mock_client_cls):
        # When token is empty, headers have no Authorization
        gh_no_token = GitHubClient(token="")
        headers = gh_no_token._headers()
        self.assertNotIn("Authorization", headers)

        # When remote API returns 401 Unauthorized
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"
        mock_response.text = '{"message": "Bad credentials"}'
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        gh_client = GitHubClient(token="invalid_token")
        res = gh_client.get_repo("test", "repo")
        self.assertIn("error", res)
        self.assertEqual(res.get("status_code"), 401)
        self.assertEqual(res.get("status"), "http_error")

    # 3. Invalid response test
    @patch("httpx.Client")
    def test_invalid_response_handling(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Malformed JSON")
        mock_response.text = "Not JSON data"
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.example.com", max_retries=0)
        res = client.get("/data")
        # Should gracefully return text fallback without crashing
        self.assertIn("text", res)
        self.assertEqual(res["text"], "Not JSON data")

        # Weather parser malformed structure handling
        wc = WeatherClient()
        with patch.object(wc, "get", return_value={"unexpected": "structure"}):
            w_res = wc.get_weather("Paris")
            self.assertEqual(w_res.get("status"), "invalid_response")
            self.assertFalse(w_res.get("success", False))

    # 4. Rate limit test
    @patch("httpx.Client")
    def test_rate_limit_handling(self, mock_client_cls):
        # 429 status code
        mock_client = MagicMock()
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_429.text = "Too Many Requests"
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_resp_429
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.github.com", max_retries=0)
        res_429 = client.get("/rate_limit")
        self.assertEqual(res_429.get("status"), "rate_limited")
        self.assertTrue(res_429.get("rate_limited"))

        # 403 status code with rate limit message
        mock_resp_403 = MagicMock()
        mock_resp_403.status_code = 403
        mock_resp_403.text = '{"message": "API rate limit exceeded for 127.0.0.1"}'
        mock_client.get.return_value = mock_resp_403

        res_403 = client.get("/repos/test/repo")
        self.assertEqual(res_403.get("status"), "rate_limited")
        self.assertTrue(res_403.get("rate_limited"))

    # 5. Secret masking test
    def test_secret_masking(self):
        sample_pat = "ghp_123456789012345678901234567890123456"
        sample_fine_pat = "github_pat_11AAAAAAA0123456789abcdef01234"
        sample_bearer = "Bearer my_super_secret_token_12345"

        masked = redact_secrets(f"Request with {sample_pat} and {sample_fine_pat} and {sample_bearer}")
        self.assertNotIn(sample_pat, masked)
        self.assertNotIn(sample_fine_pat, masked)
        self.assertNotIn("my_super_secret_token_12345", masked)

        # BaseApiClient header sanitization
        client = BaseApiClient(base_url="https://api.github.com")
        headers = {"Authorization": f"token {sample_pat}", "Accept": "application/json"}
        sanitized_headers = client._sanitize_headers(headers)
        self.assertNotIn(sample_pat, sanitized_headers.get("Authorization", ""))

    # 6. Offline behavior test
    @patch("httpx.Client")
    def test_offline_behavior(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Network unreachable")
        mock_client_cls.return_value = mock_client

        client = BaseApiClient(base_url="https://api.github.com", max_retries=0)
        res = client.get("/repos/owner/repo")
        self.assertEqual(res.get("status"), "offline")
        self.assertTrue(res.get("offline"))
        self.assertFalse(res.get("success", False))

        # AgentCore offline integration handling
        agent = AgentCore()
        resp_gh = agent.process("check my GitHub repository")
        self.assertIn("offline", resp_gh.answer.lower())
        self.assertIn("offline", resp_gh.spoken_summary.lower())

        resp_weather = agent.process("what is the weather")
        self.assertIn("offline", resp_weather.answer.lower())
        self.assertIn("offline", resp_weather.spoken_summary.lower())

    # 7. GitHub tool selection test
    def test_github_tool_selection(self):
        router = ExecutionRouter()

        # RouteType.AGENT classification
        self.assertEqual(router.classify("check my GitHub repository"), RouteType.AGENT)
        self.assertEqual(router.classify("show recent GitHub commits"), RouteType.AGENT)
        self.assertEqual(router.classify("check github status"), RouteType.AGENT)

        # AgentTaskProfile classification
        self.assertEqual(classify_agent_task_profile("check my GitHub repository"), AgentTaskProfile.GITHUB_STATUS)
        self.assertEqual(classify_agent_task_profile("show recent GitHub commits"), AgentTaskProfile.GITHUB_COMMITS)
        self.assertEqual(classify_agent_task_profile("github commits"), AgentTaskProfile.GITHUB_COMMITS)

        # Conceptual query stays INFORMATION
        self.assertEqual(router.classify("what is github"), RouteType.INFORMATION)

    # 8. Weather tool selection test
    def test_weather_tool_selection(self):
        router = ExecutionRouter()

        # RouteType.AGENT classification
        self.assertEqual(router.classify("what is the weather"), RouteType.AGENT)
        self.assertEqual(router.classify("what's the weather"), RouteType.AGENT)
        self.assertEqual(router.classify("weather in London"), RouteType.AGENT)
        self.assertEqual(router.classify("check the weather forecast"), RouteType.AGENT)

        # AgentTaskProfile classification
        self.assertEqual(classify_agent_task_profile("what is the weather"), AgentTaskProfile.WEATHER_INSPECTION)
        self.assertEqual(classify_agent_task_profile("what's the weather"), AgentTaskProfile.WEATHER_INSPECTION)
        self.assertEqual(classify_agent_task_profile("weather in Paris"), AgentTaskProfile.WEATHER_INSPECTION)

        # Conceptual query stays INFORMATION
        self.assertEqual(router.classify("what is weather forecasting"), RouteType.INFORMATION)

    # 9. Audit logging test
    def test_audit_logging_for_api_tools(self):
        audit_logger.clear()

        # Execute get_repository tool via registry
        with patch.object(GitHubClient, "get_repo", return_value={"name": "test/repo", "stars": 5, "success": True, "verified": True}):
            res = registry.execute("get_repository", {"owner": "test", "repo": "repo"}, user_request="check my github repository")
            self.assertTrue(res.success)

        events = audit_logger.get_events()
        self.assertGreaterEqual(len(events), 1)
        last_event = events[-1]
        self.assertEqual(last_event["tool"], "get_repository")
        self.assertEqual(last_event["risk_level"], "low")
        self.assertEqual(last_event["permission_level"], "read_only")
        self.assertEqual(last_event["confirmation_status"], "NOT_REQUIRED")
        self.assertTrue(last_event["success"])

    # 10. No arbitrary URL access test
    def test_no_arbitrary_url_access(self):
        client = BaseApiClient(base_url="https://api.github.com")

        # Request to unauthorized arbitrary external URL must be rejected
        res = client.get("https://attacker-site.com/steal-data")
        self.assertEqual(res.get("status"), "invalid_url")
        self.assertIn("Arbitrary URL access rejected", res.get("error", ""))

        # Verify no generic HTTP fetch / curl tool is registered in the tool registry
        registered_tool_names = registry.list_tools()
        for forbidden in ["curl", "fetch_url", "http_get", "arbitrary_request", "web_request"]:
            self.assertNotIn(forbidden, registered_tool_names)


if __name__ == "__main__":
    unittest.main()
