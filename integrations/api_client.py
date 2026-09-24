import re
import time
from typing import Dict, Any, Optional
import httpx

from config import HTTP_TIMEOUT_SECONDS
from observability.logging import get_logger, sanitize_data

logger = get_logger("CyberJarvis.ApiClient")


def redact_secrets(val: Any) -> Any:
    """Mask any authorization tokens, Bearer secrets, and GitHub PATs."""
    if isinstance(val, str):
        val = re.sub(r"(?i)(bearer|token)\s+[a-zA-Z0-9_\-\.]+", r"\1 [REDACTED]", val)
        val = re.sub(r"ghp_[a-zA-Z0-9]{30,}", "[REDACTED_GITHUB_TOKEN]", val)
        val = re.sub(r"github_pat_[a-zA-Z0-9_]{22,}", "[REDACTED_GITHUB_TOKEN]", val)
    return sanitize_data(val)


class BaseApiClient:
    """
    Robust, secure HTTP client abstraction with bounded connect/read timeouts,
    retry policy with strict bounds, error normalization, secret redaction,
    and protection against arbitrary URL redirection.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = HTTP_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = min(3.0, timeout)
        self.read_timeout = timeout
        self.max_retries = max_retries

    def _sanitize_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not headers:
            return {}
        sanitized = {}
        for k, v in headers.items():
            if k.lower() in ("authorization", "proxy-authorization", "x-api-key"):
                sanitized[k] = redact_secrets(v)
            else:
                sanitized[k] = v
        return sanitized

    def _build_url(self, endpoint: str) -> str:
        """
        Build and validate target URL.
        Prevents arbitrary external URL requests when base_url is configured.
        """
        endpoint_clean = endpoint.strip()
        if self.base_url:
            # If endpoint starts with http:// or https://, verify it matches base_url
            if endpoint_clean.startswith("http://") or endpoint_clean.startswith("https://"):
                if not endpoint_clean.startswith(self.base_url):
                    raise ValueError(f"Arbitrary URL access rejected: URL '{endpoint_clean}' does not match configured base '{self.base_url}'")
                return endpoint_clean
            return f"{self.base_url}/{endpoint_clean.lstrip('/')}"
        else:
            # When base_url is not set, only allow http(s) URLs if explicitly intended
            if not (endpoint_clean.startswith("http://") or endpoint_clean.startswith("https://")):
                raise ValueError(f"Invalid URL: '{endpoint_clean}' must include scheme when base_url is empty.")
            return endpoint_clean

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute GET request with bounded timeouts, retries, and normalized error responses.
        """
        try:
            url = self._build_url(endpoint)
        except ValueError as ex:
            logger.warning(f"URL validation failed: {ex}")
            return {"error": str(ex), "status": "invalid_url", "success": False}

        headers = headers or {}
        headers.setdefault("User-Agent", "CyberJarvis-Agent/2.0")

        # Configure bounded timeouts
        timeout_config = httpx.Timeout(
            timeout=self.timeout,
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.connect_timeout,
        )

        attempts = 0
        last_error = ""

        while attempts <= self.max_retries:
            attempts += 1
            try:
                with httpx.Client(timeout=timeout_config) as client:
                    response = client.get(url, params=params, headers=headers)

                    # 1. Rate Limit Detection (429 or 403 with rate limit message)
                    is_rate_limited = (
                        response.status_code == 429
                        or (response.status_code == 403 and any(
                            phrase in response.text.lower() for phrase in ["rate limit", "rate_limit", "too many requests"]
                        ))
                    )
                    if is_rate_limited:
                        logger.warning(f"API rate limit reached for {url} (status {response.status_code})")
                        return {
                            "error": "API rate limit exceeded. Please wait before retrying.",
                            "status": "rate_limited",
                            "status_code": response.status_code,
                            "rate_limited": True,
                            "success": False,
                        }

                    # 2. HTTP Error handling
                    if response.status_code >= 400:
                        # Only retry 5xx transient server errors
                        if response.status_code in (500, 502, 503, 504) and attempts <= self.max_retries:
                            time.sleep(0.15 * attempts)
                            continue

                        safe_err = redact_secrets(response.text[:200])
                        logger.warning(f"HTTP error {response.status_code} for GET {url}: {safe_err}")
                        return {
                            "error": f"HTTP {response.status_code}: {response.reason_phrase}",
                            "status": "http_error",
                            "status_code": response.status_code,
                            "details": safe_err,
                            "success": False,
                        }

                    # 3. Successful Response Validation
                    try:
                        return response.json()
                    except Exception:
                        return {"text": response.text, "success": True}

            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as ex:
                last_error = f"Request to {url} timed out after {self.timeout}s."
                if attempts <= self.max_retries:
                    time.sleep(0.15 * attempts)
                    continue
                logger.error(f"HTTP request timed out for {url}: {ex}")
                return {
                    "error": redact_secrets(last_error),
                    "status": "timeout",
                    "timed_out": True,
                    "success": False,
                }

            except (httpx.ConnectError, httpx.NetworkError) as ex:
                safe_msg = redact_secrets(str(ex))
                logger.error(f"Network offline / connection error for {url}: {safe_msg}")
                return {
                    "error": f"Network offline or host unreachable: {safe_msg}",
                    "status": "offline",
                    "offline": True,
                    "success": False,
                }

            except Exception as ex:
                safe_msg = redact_secrets(str(ex))
                logger.error(f"Unexpected error requesting {url}: {safe_msg}")
                return {
                    "error": f"Unexpected error: {safe_msg}",
                    "status": "error",
                    "success": False,
                }

        return {
            "error": redact_secrets(last_error or "Maximum retries exceeded"),
            "status": "retry_exhausted",
            "success": False,
        }
