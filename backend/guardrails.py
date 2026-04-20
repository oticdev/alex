"""
Enterprise guardrails (Guide 8 §4): output validation, input sanitization,
response size limits, and helpers for resilient Lambda invocation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Tuple

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Charter output — see backend/charter/templates.py
ALLOWED_CHART_TYPES = frozenset({"pie", "bar", "donut", "horizontalBar"})
MAX_CHARTS = 12
MAX_DATA_POINTS_PER_CHART = 500

SANITIZE_BLOCK_MESSAGE = "[INPUT REDACTED: potential prompt-injection pattern detected]"

DANGEROUS_PROMPT_PATTERNS = (
    "ignore previous instructions",
    "disregard all prior",
    "forget everything",
    "new instructions:",
    "\nsystem:",
    "\nassistant:",
    "you are now",
    "developer mode",
    "jailbreak",
)


def sanitize_user_input(text: str | None) -> str:
    """Reduce prompt-injection risk for free-form user text passed into agents or prompts."""
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    lower = s.lower()
    for pattern in DANGEROUS_PROMPT_PATTERNS:
        if pattern in lower:
            logger.warning("Guardrails: potential prompt-injection pattern matched: %r", pattern)
            return SANITIZE_BLOCK_MESSAGE
    return s


def sanitize_jsonable(value: Any) -> Any:
    """Recursively sanitize string leaves in JSON-like structures (e.g. analysis options)."""
    if isinstance(value, str):
        return sanitize_user_input(value)
    if isinstance(value, dict):
        return {k: sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_jsonable(v) for v in value]
    return value


def truncate_response(text: str, max_length: int = 50_000) -> str:
    """Cap oversized model output before persistence or downstream use."""
    if len(text) <= max_length:
        return text
    logger.warning(
        "Guardrails: truncating response from %s to %s characters", len(text), max_length
    )
    return text[:max_length] + "\n\n[Response truncated due to length]"


_COLOR_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _valid_data_point(_chart_type: str, point: Any, index: int) -> Tuple[bool, str]:
    if not isinstance(point, dict):
        return False, f"Chart data[{index}] must be an object"
    if "name" not in point or "value" not in point:
        return False, f"Chart data points must include 'name' and 'value' (index {index})"
    if not isinstance(point["name"], str):
        return False, f"Chart data[{index}].name must be a string"
    val = point["value"]
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False, f"Chart data[{index}].value must be a number"
    if "color" in point:
        c = point["color"]
        if not isinstance(c, str) or not _COLOR_HEX.match(c):
            return False, f"Chart data[{index}].color must be a #RRGGBB hex string if present"
    return True, ""


def validate_charts_document(data: Any) -> Tuple[bool, str]:
    """
    Validate parsed charter JSON (dict) against templates.py rules.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Root JSON value must be an object"

    if "charts" not in data:
        return False, "Missing required key 'charts'"

    charts = data["charts"]
    if not isinstance(charts, list):
        return False, "'charts' must be an array"

    if len(charts) > MAX_CHARTS:
        return False, f"Too many charts (max {MAX_CHARTS})"

    for i, chart in enumerate(charts):
        if not isinstance(chart, dict):
            return False, f"Chart {i} must be an object"
        for field in ("key", "title", "type", "description", "data"):
            if field not in chart:
                return False, f"Chart {i} missing required field '{field}'"
        if not isinstance(chart["key"], str) or not chart["key"].strip():
            return False, f"Chart {i} 'key' must be a non-empty string"
        if not isinstance(chart["title"], str):
            return False, f"Chart {i} 'title' must be a string"
        if not isinstance(chart["description"], str):
            return False, f"Chart {i} 'description' must be a string"
        ctype = chart["type"]
        if not isinstance(ctype, str) or ctype not in ALLOWED_CHART_TYPES:
            return (
                False,
                f"Chart {i} has invalid type {ctype!r}; expected one of {sorted(ALLOWED_CHART_TYPES)}",
            )
        series = chart["data"]
        if not isinstance(series, list):
            return False, f"Chart {i} 'data' must be an array"
        if len(series) > MAX_DATA_POINTS_PER_CHART:
            return False, f"Chart {i} has too many data points (max {MAX_DATA_POINTS_PER_CHART})"
        for j, point in enumerate(series):
            ok, err = _valid_data_point(ctype, point, j)
            if not ok:
                return False, f"Chart {i}: {err}"

    return True, ""


def parse_lambda_invoke_payload(response: Dict[str, Any]) -> Dict[str, Any]:
    """Read Lambda invoke response body the same way as planner agent tools."""
    result: Dict[str, Any] = dict(response)
    if isinstance(result, dict) and "statusCode" in result and "body" in result:
        body = result["body"]
        if isinstance(body, str):
            try:
                result = json.loads(body)
            except json.JSONDecodeError:
                result = {"message": body}
        else:
            result = body if isinstance(body, dict) else {"message": body}
    return result


def is_retryable_boto_client_error(exc: BaseException) -> bool:
    """True for throttling / transient Lambda API errors worth retrying."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in (
            "TooManyRequestsException",
            "ThrottlingException",
            "ServiceUnavailable",
            "RequestTimeout",
        )
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("throttl", "timeout", "rate exceed", "slow down", "service unavailable")
    )


def lambda_invoke_request_response(
    client: Any,
    function_name: str,
    payload: Dict[str, Any],
    *,
    unwrap: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous Lambda RequestResponse invoke. Set unwrap=False to inspect statusCode/body shell
    (e.g. Instrument Tagger success checks).
    """
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )
    raw = json.loads(response["Payload"].read())
    if not isinstance(raw, dict):
        return {"message": str(raw)}
    if unwrap:
        return parse_lambda_invoke_payload(raw)
    return raw
