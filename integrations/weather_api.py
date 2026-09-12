from typing import Dict, Any
from integrations.api_client import BaseApiClient
from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel


class WeatherClient(BaseApiClient):
    """Weather API client using wttr.in JSON interface (works reliably without external keys)."""

    def __init__(self):
        super().__init__(base_url="https://wttr.in")

    def get_weather(self, location: str = "") -> Dict[str, Any]:
        endpoint = f"/{location}?format=j1" if location else "/?format=j1"
        res = self.get(endpoint)
        if "error" in res:
            return res

        try:
            curr = res.get("current_condition", [{}])[0]
            nearest = res.get("nearest_area", [{}])[0]
            city = nearest.get("areaName", [{}])[0].get("value", location or "Current Location")
            country = nearest.get("country", [{}])[0].get("value", "")

            return {
                "location": f"{city}, {country}".strip(", "),
                "temperature_c": curr.get("temp_C"),
                "feels_like_c": curr.get("FeelsLikeC"),
                "weather_desc": curr.get("weatherDesc", [{}])[0].get("value"),
                "humidity": curr.get("humidity"),
                "wind_kmph": curr.get("windspeedKmph"),
            }
        except Exception as ex:
            return {"error": f"Failed to parse weather information: {ex}"}


weather_client = WeatherClient()


@registry.register(
    name="get_current_weather",
    description="Get current weather for a city or current location.",
    parameters={
        "location": {"type": "string", "description": "City name (e.g. 'London', 'Tokyo', or empty for local)", "default": ""},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_current_weather(location: str = "") -> Dict[str, Any]:
    return weather_client.get_weather(location)
