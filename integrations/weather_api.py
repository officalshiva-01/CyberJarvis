import urllib.parse
from typing import Dict, Any
from integrations.api_client import BaseApiClient
from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel
from observability.logging import get_logger

logger = get_logger("CyberJarvis.WeatherApi")


class WeatherClient(BaseApiClient):
    """Weather API client using wttr.in JSON interface (works reliably without external keys)."""

    def __init__(self):
        super().__init__(base_url="https://wttr.in")

    def get_weather(self, location: str = "") -> Dict[str, Any]:
        location_clean = location.strip()
        if location_clean:
            encoded_loc = urllib.parse.quote(location_clean)
            endpoint = f"/{encoded_loc}?format=j1"
        else:
            endpoint = "/?format=j1"

        res = self.get(endpoint)
        if isinstance(res, dict) and "error" in res:
            return res

        if not isinstance(res, dict) or "current_condition" not in res or not res.get("current_condition"):
            return {
                "error": "Invalid weather API response structure: missing 'current_condition'",
                "status": "invalid_response",
                "success": False,
            }

        try:
            curr = res.get("current_condition", [{}])[0]
            nearest = res.get("nearest_area", [{}])[0]
            city = nearest.get("areaName", [{}])[0].get("value", location_clean or "Local")
            country = nearest.get("country", [{}])[0].get("value", "")

            loc_str = f"{city}, {country}".strip(", ") if country else city
            return {
                "location": loc_str,
                "temperature_c": str(curr.get("temp_C", "")),
                "feels_like_c": str(curr.get("FeelsLikeC", "")),
                "weather_desc": curr.get("weatherDesc", [{}])[0].get("value", "Clear"),
                "humidity": str(curr.get("humidity", "")),
                "wind_kmph": str(curr.get("windspeedKmph", "")),
                "success": True,
                "verified": True,
            }
        except Exception as ex:
            logger.error(f"Failed to parse weather information: {ex}")
            return {
                "error": f"Failed to parse weather information: {ex}",
                "status": "invalid_response",
                "success": False,
            }


weather_client = WeatherClient()


@registry.register(
    name="get_current_weather",
    description="Get current weather conditions for a city or the local area.",
    parameters={
        "location": {"type": "string", "description": "City name (e.g. 'London', 'Tokyo', or empty for local)", "default": ""},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_current_weather(location: str = "") -> Dict[str, Any]:
    return weather_client.get_weather(location)
