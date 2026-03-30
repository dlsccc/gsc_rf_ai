#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
"""Minimal LLM client for field auto mapping."""

import json
from urllib import request

from service.llm_service.llm_factory import LlmFactory
from utils.log_utils import LogUtils

logger = LogUtils.get_logger('data_smart_call_llm')

DEFAULT_MODEL = "qwen-s-pro"

# System prompt for field mapping task (defined before use)
SYSTEM_PROMPT = """你是数据工程字段映射助手。你的任务是把"目标字段(modelFields)"映射到"源字段(sourceFields)"，输出严格JSON。
必须遵守：
1) 只能使用提供的字段，不得臆造字段名或key。
2) 输出格式必须是：{"mappings":{"<targetFieldName>":["<sourceFieldKey>", ...]}}
3) value 是数组，可一对多；无匹配可不返回该target。
4) 优先考虑：字段名语义 > 类型兼容 > 样例值语义。
5) 若不确定，宁可不映射，不要猜测。
6) 只输出JSON，不要Markdown，不要解释文字。"""


def chat_completion(messages, model=DEFAULT_MODEL, lang="zh-CN", is_risk_control=False, timeout=30):
    """
    Call LLM to get chat completion.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name to use (default: qwen-s-pro)
        lang: Language setting
        is_risk_control: Whether to enable risk control (not used in this implementation)
        timeout: Request timeout (not used directly, handled by RPC service)

    Returns:
        str: The text content from LLM response
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    # Convert messages to prompt format expected by LLM
    # System message + user message concatenation
    logger.info(f"chat_completion: received {len(messages)} messages")
    logger.info(f"chat_completion: messages = {messages}")

    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"[System]\n{content}")
        else:
            prompt_parts.append(content)

    prompt = "\n".join(prompt_parts)
    logger.info(f"chat_completion: prompt = {prompt[:100]}...")

    model_use = LlmFactory.get_model(model_name=model)
    result = model_use.send_request(prompt=prompt)
    return result if isinstance(result, str) else str(result)


def build_messages(model_fields, source_fields):
    """
    Build messages for field mapping task.

    Args:
        model_fields: List of target model field definitions
        source_fields: List of source field definitions

    Returns:
        list: Messages ready for LLM chat completion
    """
    payload = {
        "task": "map_fields",
        "modelFields": model_fields,
        "sourceFields": source_fields
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def extract_json_dict(text):
    """
    Extract JSON object from LLM response text.

    Args:
        text: Raw LLM response text

    Returns:
        dict: Parsed JSON object
    """
    text = str(text).strip()
    if not text:
        raise RuntimeError("llm response is empty")

    # Remove markdown code blocks if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

    # Try to find and extract JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        # Found a potential JSON object
        try:
            payload = json.loads(text[start: end + 1])
            if not isinstance(payload, dict):
                raise RuntimeError("llm response json must be object")
            return payload
        except json.JSONDecodeError:
            # Not valid JSON, fall through to error
            pass

    # If we get here, no valid JSON object was found
    # Try to parse the whole text as JSON (handles array cases)
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError("llm response json must be object")
        return payload
    except json.JSONDecodeError:
        pass

    raise RuntimeError("llm response does not contain a json object")