#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
"""Field mapping: try LLM first, fallback to rule matching."""

import re

from .call_llm import chat_completion, build_messages, extract_json_dict
from utils.log_utils import LogUtils

logger = LogUtils.get_logger('mapper')


LOCATION_TARGET_KEYWORDS = ("location", "coordinate", "position", "geo", "坐标", "经纬", "位置")
LNG_CANDIDATES = {"lng", "lon", "longitude", "long", "经度"}
LAT_CANDIDATES = {"lat", "latitude", "纬度"}
QOS_INDICATOR_FIELD_NAMES = {"nr5gqosindicator"}
FIELD_NAME_5QI_PATTERN = re.compile(r"5QI(\d+)", re.IGNORECASE)


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
                "fieldBusinessType": _to_text(item.get("fieldBusinessType") or item.get("businessType")).lower(),
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


def _find_5qi_special_mapping(model_item, source_fields):
    target_name = _to_text(model_item.get("fieldName"))
    target_desc = _to_text(model_item.get("fieldDesc"))
    business_type = _to_text(model_item.get("fieldBusinessType")).lower()
    if business_type != "metric":
        return []

    match = FIELD_NAME_5QI_PATTERN.search(target_name)
    match_source_text = target_name
    if not match and target_desc:
        match = FIELD_NAME_5QI_PATTERN.search(target_desc)
        match_source_text = target_desc
    if not match:
        return []

    target_without_number = FIELD_NAME_5QI_PATTERN.sub("5QI", match_source_text)
    normalized_without_number = _normalize_name(target_without_number)
    qos_indicator_key = ""
    base_field_key = ""

    for item in source_fields:
        normalized_name = _normalize_name(item["fieldName"])
        if not qos_indicator_key and normalized_name in QOS_INDICATOR_FIELD_NAMES:
            qos_indicator_key = item["fieldKey"]
        if not base_field_key and normalized_name == normalized_without_number:
            base_field_key = item["fieldKey"]
        if qos_indicator_key and base_field_key:
            break

    if qos_indicator_key and base_field_key:
        return [qos_indicator_key, base_field_key]
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
        special_mapping = _find_5qi_special_mapping(item, source_fields)
        if special_mapping:
            mappings[target_name] = special_mapping
            continue
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


# 匹配阈值：描述长度小于此值时不进行描述匹配（避免太短的描述误匹配）
MIN_DESC_MATCH_LENGTH = 5

# 需要排除的source字段名（这些太通用，容易误匹配）
EXCLUDED_SOURCE_NAMES = {
    "time", "date", "name", "value", "id", "type", "status", "flag", "code",
    "desc", "remark", "note", "info", "data", "num", "no", "count", "amount",
    "result", "result1", "result2", "result3", "result4",
    "time1", "time2", "time3", "time4",
    "date1", "date2", "date3", "date4",
    "val", "v1", "v2", "v3", "v4",
}


def _rule_match_by_name_and_desc(model_fields, source_fields):
    """基于字段名和字段描述的规则匹配"""
    mappings = {}

    # 构建source字段索引（排除通用字段名）
    # 注意：多表同名字段只保留第一个（主表），避免重复映射
    source_by_name = {}
    source_by_normalized_name = {}
    for item in source_fields:
        name = item["fieldName"]
        normalized_name = _normalize_name(name)
        # 跳过太通用的字段名
        if normalized_name in EXCLUDED_SOURCE_NAMES:
            continue
        key = item["fieldKey"]
        # 只保留第一个出现的字段名（主表），避免多表同名字段重复映射
        if name not in source_by_name:
            source_by_name[name] = key
        # 规范化匹配也只保留第一个
        if normalized_name not in source_by_normalized_name:
            source_by_normalized_name[normalized_name] = key

    for model_item in model_fields:
        target_name = model_item["fieldName"]
        target_desc = _normalize_name(model_item.get("fieldDesc", ""))

        special_mapping = _find_5qi_special_mapping(model_item, source_fields)
        if special_mapping:
            mappings[target_name] = special_mapping
            continue

        # 1. 精确匹配字段名
        if target_name in source_by_name:
            mappings[target_name] = [source_by_name[target_name]]
            continue

        # 2. 规范化匹配字段名
        normalized_target = _normalize_name(target_name)
        if normalized_target in source_by_normalized_name:
            mappings[target_name] = [source_by_normalized_name[normalized_target]]
            continue

        # 3. 匹配字段描述（需要描述长度足够长）
        if len(target_desc) >= MIN_DESC_MATCH_LENGTH:
            # 精确匹配：source字段名 去除空格/下划线/横线 后 与 model描述完全一致
            if target_desc in source_by_normalized_name:
                mappings[target_name] = [source_by_normalized_name[target_desc]]
                continue

    return mappings


# 每批处理的字段数量，可根据大模型输入限制调整
DEFAULT_MODEL_BATCH_SIZE = 300  # modelFields 每批数量
DEFAULT_SOURCE_BATCH_SIZE = 50  # sourceFields 每批数量


def _calculate_batches(model_fields, source_fields, model_batch_size, source_batch_size):
    """计算批次信息"""
    model_batches = (len(model_fields) + model_batch_size - 1) // model_batch_size
    source_batches = (len(source_fields) + source_batch_size - 1) // source_batch_size
    total_calls = model_batches * source_batches
    return model_batches, source_batches, total_calls


def _process_batch(chat_fn, model_fields, source_fields, model_batch_size, source_batch_size):
    """执行双层分批调用LLM，返回映射结果"""
    llm_mappings = {}
    llm_error = ""
    model_batches, source_batches, total_calls = _calculate_batches(
        model_fields, source_fields, model_batch_size, source_batch_size)

    for model_idx in range(model_batches):
        model_start = model_idx * model_batch_size
        model_end = min((model_idx + 1) * model_batch_size, len(model_fields))
        batch_model = model_fields[model_start:model_end]

        for source_idx in range(source_batches):
            source_start = source_idx * source_batch_size
            source_end = min((source_idx + 1) * source_batch_size, len(source_fields))
            batch_source = source_fields[source_start:source_end]

            try:
                messages = build_messages(batch_model, batch_source)
                call_idx = model_idx * source_batches + source_idx + 1
                logger.info(f"call {call_idx}/{total_calls} - model[{model_start}:{model_end}] + source[{source_start}:{source_end}]")

                llm_text = chat_fn(messages)
                llm_payload = extract_json_dict(llm_text)
                batch_mappings = _sanitize_mappings(llm_payload, batch_model, batch_source)

                # 合并结果
                for target_name, source_keys in batch_mappings.items():
                    if target_name not in llm_mappings:
                        llm_mappings[target_name] = []
                    for key in source_keys:
                        if key not in llm_mappings[target_name]:
                            llm_mappings[target_name].append(key)

            except Exception as exc:  # noqa: BLE001
                call_idx = model_idx * source_batches + source_idx + 1
                logger.error(f"call {call_idx} failed: {exc}")
                llm_error = f"llm_failed:{exc.__class__.__name__}"

    return llm_mappings, llm_error


def _merge_results(model_fields, llm_mappings, fallback_mappings):
    """合并LLM结果和规则匹配结果"""
    merged = {}
    fallback_applied = False
    for field in model_fields:
        name = field["fieldName"]
        if name in llm_mappings:
            merged[name] = llm_mappings[name]
        elif name in fallback_mappings:
            merged[name] = fallback_mappings[name]
            fallback_applied = True
    return merged, fallback_applied


def auto_map_fields(model_fields, source_fields, llm_chat_fn=None,
                    model_batch_size=DEFAULT_MODEL_BATCH_SIZE,
                    source_batch_size=DEFAULT_SOURCE_BATCH_SIZE,
                    use_llm=False):
    """Return mapping result dict: mappings + fallback info.

    双层分批处理：modelFields和sourceFields都分批，
    每批组合都带上足够的上下文信息。

    Args:
        model_fields: 目标模型字段列表
        source_fields: 源字段列表
        llm_chat_fn: 可选的LLM调用函数
        model_batch_size: modelFields每批数量
        source_batch_size: sourceFields每批数量
        use_llm: 是否使用LLM匹配，默认False（仅使用规则匹配）
    """
    model_fields = _clean_model_fields(model_fields)
    source_fields = _clean_source_fields(source_fields)

    if not model_fields or not source_fields:
        return {"mappings": {}, "fallbackApplied": False, "fallbackReason": ""}

    logger.info(f"auto_map_fields: {len(model_fields)} model + {len(source_fields)} source fields, use_llm={use_llm}")

    # 规则匹配（基于字段名和字段描述）
    rule_mappings = _rule_match_by_name_and_desc(model_fields, source_fields)
    logger.info(f"auto_map_fields: rule matching found {len(rule_mappings)} mappings")

    # 如果不使用LLM，直接返回规则匹配结果
    if not use_llm:
        return {
            "mappings": rule_mappings,
            "fallbackApplied": False,
            "fallbackReason": ""
        }

    # 计算需要的批次数
    model_batches, source_batches, total_calls = _calculate_batches(
        model_fields, source_fields, model_batch_size, source_batch_size)

    logger.info(f"auto_map_fields: {model_batches} x {source_batches} = {total_calls} LLM calls")

    # 执行分批调用
    chat_fn = llm_chat_fn or chat_completion
    llm_mappings, llm_error = _process_batch(
        chat_fn, model_fields, source_fields, model_batch_size, source_batch_size)

    # 使用规则匹配作为回退
    fallback_mappings = _rule_fallback(model_fields, source_fields)

    # 合并结果
    if llm_mappings:
        merged, fallback_applied = _merge_results(model_fields, llm_mappings, fallback_mappings)
        return {
            "mappings": merged,
            "fallbackApplied": fallback_applied,
            "fallbackReason": "fallback_applied:partial" if fallback_applied else llm_error,
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


def _auto_map_fields_single(model_fields, source_fields, llm_chat_fn):
    """单次调用LLM的内部方法"""
    llm_mappings = {}
    llm_error = ""
    try:
        chat_fn = llm_chat_fn or chat_completion
        messages = build_messages(model_fields, source_fields)

        llm_text = chat_fn(messages)

        logger.info(f"auto_map_fields: LLM returned: {llm_text[:200] if llm_text else 'empty'}")
        llm_payload = extract_json_dict(llm_text)
        llm_mappings = _sanitize_mappings(llm_payload, model_fields, source_fields)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"auto_map_fields: LLM call failed: {exc}")
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
            "fallbackReason": "fallback_applied:partial" if fallback_applied else llm_error,
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
