#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal LLM client for field auto mapping."""

import json
import time

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

# Keep hardcoded by current project requirement.
TOKEN_CREDENTIALS = {
    "beta": {"app_key": "get-his-token", "app_secret": "MK8JDz5lwUQMMLAoDC2sqaGdThKZwHizGe3i14eh"},
    "prod": {"app_key": "get-his-token", "app_secret": "B10U5a0DsEqZX33sD7PApyZgI14EBwel8ZuTo8xu"},
}

MODEL_URL = "https://console.his.huawei.com/agi/agi_agent/infer/v1/chat/completions"
DEFAULT_ENV = "prod"
DEFAULT_APP_ID = "com.huawei.gtsfi.costline"
DEFAULT_MODEL = "GLM-V4.7"


def _to_json_obj(response, error_prefix):
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{error_prefix}: response is not valid json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{error_prefix}: response json must be object")
    return payload


def get_dynamics_token(env=DEFAULT_ENV, appid=DEFAULT_APP_ID, timeout=15):
    if env not in TOKEN_URLS:
        raise ValueError(f"unsupported env: {env}")

    token_urls = TOKEN_URLS[env]
    credentials = TOKEN_CREDENTIALS[env]

    try:
        manas_resp = requests.post(
            token_urls["manas"],
            json=credentials,
            verify=False,
            timeout=timeout,
        )
        manas_resp.raise_for_status()
        manas_payload = _to_json_obj(manas_resp, "manas token request failed")
    except requests.RequestException as exc:
        raise RuntimeError(f"manas token request failed: {exc}") from exc

    access_token = manas_payload.get("AccessToken")
    if not access_token:
        raise RuntimeError("manas token request failed: missing AccessToken")

    headers = {"AccessToken": access_token, "Content-Type": "application/json"}
    try:
        dynamic_resp = requests.post(
            token_urls["dynamic"],
            headers=headers,
            json={"appid": appid},
            verify=False,
            timeout=timeout,
        )
        dynamic_resp.raise_for_status()
        dynamic_payload = _to_json_obj(dynamic_resp, "dynamic token request failed")
    except requests.RequestException as exc:
        raise RuntimeError(f"dynamic token request failed: {exc}") from exc

    token = dynamic_payload.get("result")
    if not token:
        raise RuntimeError("dynamic token request failed: missing result token")
    return str(token)


def _extract_content(payload):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("model response missing choices")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_list = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_list.append(item["text"])
        if text_list:
            return "".join(text_list)
    raise RuntimeError("model response missing message.content")


def chat_completion(
    messages,
    model=DEFAULT_MODEL,
    env=DEFAULT_ENV,
    appid=DEFAULT_APP_ID,
    timeout=30,
    max_retries=1,
):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    body = {
        "messages": messages,
        "model": model,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    attempts = max(1, int(max_retries) + 1)
    last_error = None

    for attempt in range(attempts):
        try:
            headers = {
                "content-type": "application/json",
                "Authorization": get_dynamics_token(env=env, appid=appid, timeout=timeout),
            }
            response = requests.post(
                MODEL_URL,
                headers=headers,
                data=json.dumps(body, ensure_ascii=False),
                verify=False,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = _to_json_obj(response, "model request failed")
            return _extract_content(payload)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
                continue

    raise RuntimeError(f"model request failed: {last_error}")

