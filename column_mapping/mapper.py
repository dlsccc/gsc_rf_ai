#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field mapping: try LLM first, fallback to rule matching."""

import json
import re

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


def _to_text(value):
    return str(value or "").strip()


def _normalize_name(name):
    return re.sub(r"[\s_\-]+", "", _to_text(name)).lower()


def _unique_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _clean_model_fields(model_fields):
    if model_fields is None or not isinstance(model_fields, list):
        raise ValueError("modelFields must be a list")
    result = []
    seen = set()
    for idx, item in enumerate(model_fields):
        if not isinstance(item, dict):
            raise ValueError(f"modelFields[{idx}] must be an object")
        field_name = _to_text(item.get("fieldName"))
        if not field_name:
            raise ValueError(f"modelFields[{idx}].fieldName is required")
        if field_name in seen:
            continue
        seen.add(field_name)
        result.append(
            {
                "fieldName": field_name,
                "fieldType": _to_text(item.get("fieldType")),
                "fieldDesc": _to_text(item.get("fieldDesc")),
            }
        )
    return result


def _clean_source_fields(source_fields):
    if source_fields is None or not isinstance(source_fields, list):
        raise ValueError("sourceFields must be a list")
    result = []
    seen = set()
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
        if field_key in seen:
            continue
        seen.add(field_key)
        result.append(
            {
                "fieldKey": field_key,
                "fieldName": field_name,
                "sourceTable": source_table,
                "fieldType": _to_text(item.get("fieldType")),
                "sampleValue": _to_text(item.get("sampleValue")),
            }
        )
    return result


def _build_messages(model_fields, source_fields):
    payload = {"task": "map_fields", "modelFields": model_fields, "sourceFields": source_fields}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _extract_json_dict(text):
    text = _to_text(text)
    if not text:
        raise RuntimeError("llm response is empty")

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
        raise RuntimeError("llm response does not contain a json object")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("llm response json must be object")
    return payload


def _sanitize_mappings(raw_payload, model_fields, source_fields):
    raw_mappings = raw_payload.get("mappings", raw_payload)
    if not isinstance(raw_mappings, dict):
        return {}

    target_order = [item["fieldName"] for item in model_fields]
    source_order_map = {item["fieldKey"]: idx for idx, item in enumerate(source_fields)}
    valid_targets = set(target_order)
    valid_sources = set(source_order_map.keys())
    cleaned = {}

    for target_name, source_keys in raw_mappings.items():
        target_name = _to_text(target_name)
        if target_name not in valid_targets:
            continue

        if isinstance(source_keys, str):
            source_keys = [source_keys]
        if not isinstance(source_keys, list):
            continue

        keys = _unique_keep_order([_to_text(key) for key in source_keys if _to_text(key) in valid_sources])
        if not keys:
            continue
        keys.sort(key=lambda key: source_order_map[key])
        cleaned[target_name] = keys

    ordered = {}
    for target_name in target_order:
        if target_name in cleaned:
            ordered[target_name] = cleaned[target_name]
    return ordered


def _is_location_target(field_name):
    normalized = _normalize_name(field_name)
    return any(keyword in normalized for keyword in LOCATION_TARGET_KEYWORDS)


def _find_lon_lat(source_fields):
    lon_key = ""
    lat_key = ""
    for item in source_fields:
        normalized_name = _normalize_name(item["fieldName"])
        if not lon_key and normalized_name in LNG_CANDIDATES:
            lon_key = item["fieldKey"]
        elif not lat_key and normalized_name in LAT_CANDIDATES:
            lat_key = item["fieldKey"]
    if lon_key and lat_key:
        return [lon_key, lat_key]
    return []


def _rule_fallback(model_fields, source_fields):
    by_name = {}
    by_normalized_name = {}

    for item in source_fields:
        name = item["fieldName"]
        key = item["fieldKey"]
        by_name.setdefault(name, []).append(key)
        by_normalized_name.setdefault(_normalize_name(name), []).append(key)

    mappings = {}
    for item in model_fields:
        target_name = item["fieldName"]
        if target_name in by_name:
            mappings[target_name] = _unique_keep_order(by_name[target_name])
            continue

        normalized_name = _normalize_name(target_name)
        if normalized_name in by_normalized_name:
            mappings[target_name] = _unique_keep_order(by_normalized_name[normalized_name])
            continue

        if _is_location_target(target_name):
            lon_lat = _find_lon_lat(source_fields)
            if lon_lat:
                mappings[target_name] = lon_lat
    return mappings


def auto_map_fields(model_fields, source_fields, llm_chat_fn=None):
    """Return mapping result dict: mappings + fallback info."""
    model_fields = _clean_model_fields(model_fields)
    source_fields = _clean_source_fields(source_fields)

    if not model_fields or not source_fields:
        return {"mappings": {}, "fallbackApplied": False, "fallbackReason": ""}

    llm_mappings = {}
    llm_error = ""
    try:
        chat_fn = llm_chat_fn or chat_completion
        llm_text = chat_fn(_build_messages(model_fields, source_fields))
        llm_payload = _extract_json_dict(llm_text)
        llm_mappings = _sanitize_mappings(llm_payload, model_fields, source_fields)
    except Exception as exc:  # noqa: BLE001
        llm_error = f"llm_failed:{exc.__class__.__name__}"

    fallback_mappings = _rule_fallback(model_fields, source_fields)

    if llm_mappings:
        merged = {}
        fallback_applied = False
        for field in model_fields:
            name = field["fieldName"]
            if name in llm_mappings:
                merged[name] = llm_mappings[name]
            elif name in fallback_mappings:
                merged[name] = fallback_mappings[name]
                fallback_applied = True
        return {
            "mappings": merged,
            "fallbackApplied": fallback_applied,
            "fallbackReason": "fallback_applied:partial" if fallback_applied else "",
        }

    if fallback_mappings:
        return {
            "mappings": fallback_mappings,
            "fallbackApplied": True,
            "fallbackReason": llm_error or "fallback_applied:llm_empty_or_invalid",
        }

    return {
        "mappings": {},
        "fallbackApplied": bool(llm_error),
        "fallbackReason": llm_error,
    }

