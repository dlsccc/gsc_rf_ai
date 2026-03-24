#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""LLM client helpers used by the field auto-mapping service.

NOTE:
- Credentials are intentionally hardcoded to match current project constraints.
- This file provides reusable functions and has no import-time network side effects.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import urllib3

urllib3.disable_warnings()

TOKEN_URLS = {
    "beta": {
        "manas": "https://gzmirror-beta.manas.huawei.com/apigovernance/api/oauth/tokenByAkSk",
        "dynamic": "https://gzmirror-beta.manas.huawei.com/api/runtime/appToken/getRestAppDynamicToken",
    },
    "prod": {
        "manas": "https://gzmirror.manas.huawei.com/apigovernance/api/oauth/tokenByAkSk",
        "dynamic": "https://gzmirror.manas.huawei.com/api/runtime/appToken/getRestAppDynamicToken",
    },
}

# Kept as hardcoded values by explicit requirement.
TOKEN_CREDENTIALS = {
    "beta": {"app_key": "get-his-token", "app_secret": "MK8JDz5lwUQMMLAoDC2sqaGdThKZwHizGe3i14eh"},
    "prod": {"app_key": "get-his-token", "app_secret": "B10U5a0DsEqZX33sD7PApyZgI14EBwel8ZuTo8xu"},
}

MODEL_URL = "https://console.his.huawei.com/agi/agi_agent/infer/v1/chat/completions"
DEFAULT_ENV = "prod"
DEFAULT_APP_ID = "com.huawei.gtsfi.costline"
DEFAULT_MODEL = "GLM-V4.7"


class LLMClientError(RuntimeError):
    """Base exception for LLM client errors."""


class LLMAuthError(LLMClientError):
    """Raised when token acquisition/authentication fails."""


class LLMNetworkError(LLMClientError):
    """Raised when network request fails."""


class LLMResponseFormatError(LLMClientError):
    """Raised when model response cannot be parsed as expected."""


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMResponseFormatError("response body is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LLMResponseFormatError("response JSON must be an object")
    return payload


def _validate_env(env: str) -> None:
    if env not in TOKEN_URLS:
        raise ValueError(f"unsupported env: {env}")


def get_dynamics_token(env: str, appid: str, timeout: float = 15.0) -> str:
    """Get dynamic app token from manas + app token APIs."""
    _validate_env(env)

    token_urls = TOKEN_URLS[env]
    token_data = TOKEN_CREDENTIALS[env]

    try:
        manas_resp = requests.post(
            url=token_urls["manas"],
            json=token_data,
            verify=False,
            timeout=timeout,
        )
        manas_resp.raise_for_status()
        manas_payload = _safe_json(manas_resp)
    except requests.RequestException as exc:
        raise LLMNetworkError(f"failed to request manas token: {exc}") from exc

    manas_token = manas_payload.get("AccessToken")
    if not manas_token:
        raise LLMAuthError("missing AccessToken in manas response")

    header = {"AccessToken": manas_token, "Content-Type": "application/json"}
    data = {"appid": appid}

    try:
        dynamic_resp = requests.post(
            url=token_urls["dynamic"],
            headers=header,
            json=data,
            verify=False,
            timeout=timeout,
        )
        dynamic_resp.raise_for_status()
        dynamic_payload = _safe_json(dynamic_resp)
    except requests.RequestException as exc:
        raise LLMNetworkError(f"failed to request dynamic token: {exc}") from exc

    token = dynamic_payload.get("result")
    if not token:
        raise LLMAuthError("missing result token in dynamic token response")
    return str(token)


def build_headers(env: str = DEFAULT_ENV, appid: str = DEFAULT_APP_ID, timeout: float = 15.0) -> dict[str, str]:
    """Build model endpoint headers with fresh dynamic token."""
    return {
        "content-type": "application/json",
        "Authorization": get_dynamics_token(env=env, appid=appid, timeout=timeout),
    }


def _extract_text_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseFormatError("missing choices in model response")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMResponseFormatError("missing message in first choice")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "".join(chunks)

    raise LLMResponseFormatError("unsupported content format in model response")


def chat_completion(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    env: str = DEFAULT_ENV,
    appid: str = DEFAULT_APP_ID,
    timeout: float = 30.0,
    max_retries: int = 1,
) -> str:
    """Call chat completions endpoint and return plain content text."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    body = {
        "messages": messages,
        "model": model,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        try:
            headers = build_headers(env=env, appid=appid, timeout=timeout)
            response = requests.post(
                MODEL_URL,
                headers=headers,
                data=json.dumps(body, ensure_ascii=False),
                verify=False,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = _safe_json(response)
            return _extract_text_content(payload)
        except (requests.RequestException, LLMClientError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
                continue

    if isinstance(last_error, LLMClientError):
        raise last_error
    raise LLMNetworkError(f"model request failed: {last_error}") from last_error


if __name__ == "__main__":
    demo_messages = [{"role": "user", "content": "hi"}]
    print(chat_completion(messages=demo_messages))
