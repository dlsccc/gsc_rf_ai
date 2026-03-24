#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core auto-mapping logic for model fields and source fields."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

try:
    from .call_llm import chat_completion
except ImportError:  # pragma: no cover
    from call_llm import chat_completion


SYSTEM_PROMPT = """你是数据工程字段映射助手。你的任务是把“目标字段(modelFields)”映射到“源字段(sourceFields)”，输出严格JSON。
必须遵守：
1) 只能使用提供的字段，不得臆造字段名或key。
2) 输出格式必须是：{"mappings":{"<targetFieldName>":["<sourceFieldKey>", ...]}}
3) value 是数组，可一对多；无匹配可不返回该target。
4) 优先考虑：字段名语义 > 类型兼容 > 样例值语义。
5) 若不确定，宁可不映射，不要猜测。
6) 只输出JSON，不要Markdown，不要解释文字。"""

LOCATION_TARGET_KEYWORDS = ("location", "coordinate", "position", "geo", "坐标", "经纬", "位置")
LNG_CANDIDATES = {"lng", "lon", "longitude", "long", "经度"}
LAT_CANDIDATES = {"lat", "latitude", "纬度"}


@dataclass(frozen=True)
class AutoMapResult:
    mappings: dict[str, list[str]]
    fallback_applied: bool
    fallback_reason: str = ""


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", _to_text(name)).lower()


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _validate_and_normalize_model_fields(model_fields: Any) -> list[dict[str, str]]:
    if model_fields is None:
        raise ValueError("modelFields is required")
    if not isinstance(model_fields, list):
        raise ValueError("modelFields must be a list")

    normalized: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(model_fields):
        if not isinstance(item, dict):
            raise ValueError(f"modelFields[{idx}] must be an object")
        field_name = _to_text(item.get("fieldName"))
        if not field_name:
            raise ValueError(f"modelFields[{idx}].fieldName is required")
        if field_name in seen_names:
            continue
        seen_names.add(field_name)
        normalized.append(
            {
                "fieldName": field_name,
                "fieldType": _to_text(item.get("fieldType")),
                "fieldDesc": _to_text(item.get("fieldDesc")),
            }
        )
    return normalized


def _validate_and_normalize_source_fields(source_fields: Any) -> list[dict[str, str]]:
    if source_fields is None:
        raise ValueError("sourceFields is required")
    if not isinstance(source_fields, list):
        raise ValueError("sourceFields must be a list")

    normalized: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for idx, item in enumerate(source_fields):
        if not isinstance(item, dict):
            raise ValueError(f"sourceFields[{idx}] must be an object")
        field_key = _to_text(item.get("fieldKey"))
        field_name = _to_text(item.get("fieldName"))
        source_table = _to_text(item.get("sourceTable"))
        if not field_key:
            raise ValueError(f"sourceFields[{idx}].fieldKey is required")
        if not field_name:
            raise ValueError(f"sourceFields[{idx}].fieldName is required")
        if not source_table:
            raise ValueError(f"sourceFields[{idx}].sourceTable is required")
        if field_key in seen_keys:
            continue
        seen_keys.add(field_key)
        normalized.append(
            {
                "fieldKey": field_key,
                "fieldName": field_name,
                "sourceTable": source_table,
                "fieldType": _to_text(item.get("fieldType")),
                "sampleValue": _to_text(item.get("sampleValue")),
            }
        )
    return normalized


def validate_input(model_fields: Any, source_fields: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return _validate_and_normalize_model_fields(model_fields), _validate_and_normalize_source_fields(source_fields)


def build_prompt(model_fields: list[dict[str, str]], source_fields: list[dict[str, str]]) -> list[dict[str, str]]:
    payload = {"task": "map_fields", "modelFields": model_fields, "sourceFields": source_fields}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _extract_json_text(raw_text: str) -> str:
    text = _to_text(raw_text)
    if not text:
        raise ValueError("empty llm response")

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("llm response does not contain a valid JSON object")
    return text[start : end + 1]


def parse_and_sanitize_llm_output(
    llm_text: str,
    model_fields: list[dict[str, str]],
    source_fields: list[dict[str, str]],
) -> dict[str, list[str]]:
    json_text = _extract_json_text(llm_text)
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("llm response JSON must be an object")

    raw_mappings = payload.get("mappings", payload)
    if not isinstance(raw_mappings, dict):
        raise ValueError("llm response must include object mappings")

    target_order = [item["fieldName"] for item in model_fields]
    valid_targets = set(target_order)
    source_order = {item["fieldKey"]: idx for idx, item in enumerate(source_fields)}

    normalized: dict[str, list[str]] = {}
    for target_name, source_keys in raw_mappings.items():
        target_text = _to_text(target_name)
        if target_text not in valid_targets:
            continue

        if isinstance(source_keys, str):
            source_key_list = [source_keys]
        elif isinstance(source_keys, list):
            source_key_list = [_to_text(item) for item in source_keys]
        else:
            continue

        filtered = [key for key in _ordered_unique(source_key_list) if key in source_order]
        if not filtered:
            continue
        filtered.sort(key=lambda key: source_order[key])
        normalized[target_text] = filtered

    ordered: dict[str, list[str]] = {}
    for target_name in target_order:
        if target_name in normalized:
            ordered[target_name] = normalized[target_name]
    return ordered


def _is_location_target(target_name: str) -> bool:
    normalized = _normalize_name(target_name)
    return any(keyword in normalized for keyword in LOCATION_TARGET_KEYWORDS)


def _find_lon_lat_pair(source_fields: list[dict[str, str]]) -> list[str]:
    lon_key = ""
    lat_key = ""
    for source in source_fields:
        normalized_name = _normalize_name(source["fieldName"])
        if not lon_key and normalized_name in LNG_CANDIDATES:
            lon_key = source["fieldKey"]
        elif not lat_key and normalized_name in LAT_CANDIDATES:
            lat_key = source["fieldKey"]
    if lon_key and lat_key:
        return [lon_key, lat_key]
    return []


def fallback_if_needed(model_fields: list[dict[str, str]], source_fields: list[dict[str, str]]) -> dict[str, list[str]]:
    sources_by_exact_name: dict[str, list[str]] = {}
    sources_by_normalized_name: dict[str, list[str]] = {}

    for source in source_fields:
        source_name = source["fieldName"]
        normalized_name = _normalize_name(source_name)
        sources_by_exact_name.setdefault(source_name, []).append(source["fieldKey"])
        sources_by_normalized_name.setdefault(normalized_name, []).append(source["fieldKey"])

    mappings: dict[str, list[str]] = {}
    for model_field in model_fields:
        target_name = model_field["fieldName"]

        exact_matches = sources_by_exact_name.get(target_name, [])
        if exact_matches:
            mappings[target_name] = _ordered_unique(exact_matches)
            continue

        normalized_matches = sources_by_normalized_name.get(_normalize_name(target_name), [])
        if normalized_matches:
            mappings[target_name] = _ordered_unique(normalized_matches)
            continue

        if _is_location_target(target_name):
            lon_lat_pair = _find_lon_lat_pair(source_fields)
            if lon_lat_pair:
                mappings[target_name] = lon_lat_pair
    return mappings


def _merge_with_fallback(
    llm_mappings: dict[str, list[str]],
    fallback_mappings: dict[str, list[str]],
    model_fields: list[dict[str, str]],
) -> tuple[dict[str, list[str]], bool]:
    merged: dict[str, list[str]] = {}
    fallback_applied = False

    for model_field in model_fields:
        target_name = model_field["fieldName"]
        if target_name in llm_mappings:
            merged[target_name] = llm_mappings[target_name]
            continue
        if target_name in fallback_mappings:
            merged[target_name] = fallback_mappings[target_name]
            fallback_applied = True
    return merged, fallback_applied


def auto_map_fields(
    model_fields: Any,
    source_fields: Any,
    llm_chat_fn: Callable[[list[dict[str, str]]], str] | None = None,
) -> AutoMapResult:
    validated_model_fields, validated_source_fields = validate_input(model_fields, source_fields)
    if not validated_model_fields or not validated_source_fields:
        return AutoMapResult(mappings={}, fallback_applied=False, fallback_reason="")

    chat_fn = llm_chat_fn or (lambda messages: chat_completion(messages=messages))
    prompt_messages = build_prompt(validated_model_fields, validated_source_fields)

    llm_mappings: dict[str, list[str]] = {}
    llm_error = ""
    try:
        llm_text = chat_fn(prompt_messages)
        llm_mappings = parse_and_sanitize_llm_output(llm_text, validated_model_fields, validated_source_fields)
    except Exception as exc:  # noqa: BLE001
        llm_error = f"llm_error:{exc.__class__.__name__}"

    fallback_mappings = fallback_if_needed(validated_model_fields, validated_source_fields)

    if llm_mappings:
        merged, fallback_applied = _merge_with_fallback(llm_mappings, fallback_mappings, validated_model_fields)
        reason = "fallback_applied:partial" if fallback_applied else ""
        return AutoMapResult(mappings=merged, fallback_applied=fallback_applied, fallback_reason=reason)

    if fallback_mappings:
        reason = llm_error or "fallback_applied:llm_empty_or_invalid"
        return AutoMapResult(mappings=fallback_mappings, fallback_applied=True, fallback_reason=reason)

    if llm_error:
        return AutoMapResult(mappings={}, fallback_applied=True, fallback_reason=llm_error)
    return AutoMapResult(mappings={}, fallback_applied=False, fallback_reason="")

