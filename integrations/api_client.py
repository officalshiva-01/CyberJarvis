import httpx
from typing import Dict, Any, Optional
from config import HTTP_TIMEOUT_SECONDS
from observability.logging import get_logger

logger = get_logger("CyberJarvis.ApiClient")


class BaseApiClient:
    """Robust base HTTP client with timeout, error handling and status checks."""

    def __init__(self, base_url: str = "", timeout: float = HTTP_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint
        headers = headers or {}
        headers.setdefault("User-Agent", "CyberJarvis-Agent/2.0")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params, headers=headers)
                if response.status_code >= 400:
                    logger.warning(f"HTTP error {response.status_code} for GET {url}: {response.text[:200]}")
                    return {
                        "error": f"HTTP {response.status_code}: {response.reason_phrase}",
                        "status_code": response.status_code,
                    }
                try:
                    return response.json()
                except Exception:
                    return {"text": response.text}
        except httpx.TimeoutException:
            logger.error(f"HTTP request timed out for {url}")
            return {"error": f"Request to {url} timed out after {self.timeout}s."}
        except httpx.RequestError as ex:
            logger.error(f"Network error requesting {url}: {ex}")
            return {"error": f"Network error: {str(ex)}"}
        except Exception as ex:
            logger.error(f"Unexpected error requesting {url}: {ex}")
            return {"error": f"Unexpected error: {str(ex)}"}
