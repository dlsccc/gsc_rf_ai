#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.


import re

from .call_llm import chat_completion, build_messages, extract_json_dict
from utils.log_utils import LogUtils

logger = LogUtils.get_logger('process_generation_mapper')

FILTER_OPERATORS = {
    "equals", "not_equals", "contains", "is_empty", "is_not_empty", "greater_than", "less_than"
}
TIME_TRANSFORM_INPUT_TYPES = {"format_datetime", "extract_year", "extract_month", "extract_time", "format_time"}
TRANSFORM_TYPES = {"format_datetime", "calc_week", "calc_weekday", "set_value", "concat", "replace"}

DEFAULT_MAPPING_BATCH_SIZE = 50

TIME_TOKEN_PATTERN = re.compile(r"(YYYY|MM|DD|hh|mm|ss)")
TIME_KEYWORDS = (
    "time", "date", "year", "month", "day", "week", "hour", "minute", "second",
    "时间", "日期", "年", "月", "日", "周", "星期", "时", "分", "秒"
)
NON_NUMERIC_TOKEN_BLACKLIST = {
    "nan", "nil", "null", "none", "na", "n/a", "-", "--", "", " ", "unk", "unknown", "无", "空"
}


def _to_text(value):
    return str(value or "").strip()


def _to_bool(value):
    if isinstance(value, bool):
        return value
    text = _to_text(value).lower()
    if text in {"true", "1", "y", "yes"}:
        return True
    if text in {"false", "0", "n", "no"}:
        return False
    return bool(value)


def _unique_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_transform_type(raw_type):
    transform_type = _to_text(raw_type).lower()
    if transform_type in TIME_TRANSFORM_INPUT_TYPES:
        return "format_datetime"
    return transform_type

def _clean_model_fields_legacy(model_fields):
    if model_fields is None or not isinstance(model_fields, list):
        raise ValueError("modelFields must be a list")

    result = []
    seen = set()
    for idx, item in enumerate(model_fields):
        if not isinstance(item, dict):
            raise ValueError("modelFields[{}] must be an object".format(idx))

        field_name = _to_text(item.get("fieldName"))
        if not field_name:
            raise ValueError("modelFields[{}].fieldName is required".format(idx))
        if field_name in seen:
            continue
        seen.add(field_name)

        result.append({
            "fieldName": field_name,
            "fieldType": _to_text(item.get("fieldType")),
            "businessDesc": _to_text(item.get("businessDesc") or item.get("fieldDesc") or item.get("description")),
            "sampleValue": _to_text(item.get("sampleValue") or item.get("example")),
            "businessType": _to_text(item.get("businessType") or item.get("fieldBusinessType")).lower(),
            "targetFormat": _to_text(item.get("targetFormat") or item.get("format") or item.get("dataFormat")),
            "modelType": _to_text(item.get("modelType") or item.get("businessModelType")).lower(),
            "involveCalc": _to_bool(item.get("involveCalc"))
        })

    return result


def _clean_model_detail_list_input(model_fields):
    """处理列表输入格式"""
    model_fields = _clean_model_fields_legacy(model_fields)
    return {
        "code": "",
        "modelName": "",
        "businessModelType": "",
        "involveCalc": False
    }, model_fields


def _clean_model_detail_legacy(model_detail):
    """处理旧格式modelFields输入"""
    model_fields = _clean_model_fields_legacy(model_detail.get("modelFields"))
    return {
        "code": _to_text(model_detail.get("code") or model_detail.get("modelCode")),
        "modelName": _to_text(model_detail.get("modelName") or model_detail.get("name")),
        "businessModelType": _to_text(
            model_detail.get("businessModelType") or model_detail.get("modelType") or model_detail.get("type")
        ).lower(),
        "involveCalc": _to_bool(model_detail.get("involveCalc"))
    }, model_fields


def _build_model_field_item(item, idx, model_type, involve_calc):
    """构建单个模型字段"""
    field_name = _to_text(item.get("fieldName") or item.get("name"))
    if not field_name:
        raise ValueError("fieldList[{}].fieldName is required".format(idx))
    return {
        "fieldName": field_name,
        "fieldType": _to_text(item.get("fieldType") or item.get("type")),
        "businessDesc": _to_text(item.get("fieldDesc") or item.get("description")),
        "sampleValue": _to_text(item.get("dataExample") or item.get("sampleValue") or item.get("example")),
        "businessType": _to_text(item.get("fieldBusinessType") or item.get("businessType")).lower(),
        "targetFormat": _to_text(item.get("dataFormat") or item.get("targetFormat") or item.get("format")),
        "modelType": _to_text(item.get("modelType") or item.get("businessModelType") or model_type).lower(),
        "involveCalc": _to_bool(item.get("involveCalc") if item.get("involveCalc") is not None else involve_calc)
    }


def _clean_model_detail(model_detail):
    if isinstance(model_detail, list):
        return _clean_model_detail_list_input(model_detail)

    if model_detail is None or not isinstance(model_detail, dict):
        raise ValueError("model detail must be an object")

    # 兼容旧入参：仍允许直接传 modelFields
    if isinstance(model_detail.get("modelFields"), list) and model_detail.get("fieldList") is None:
        return _clean_model_detail_legacy(model_detail)

    field_list = model_detail.get("fieldList")
    if field_list is None or not isinstance(field_list, list):
        raise ValueError("fieldList must be a list")

    model_type = _to_text(
        model_detail.get("businessModelType") or model_detail.get("modelType") or model_detail.get("type")
    ).lower()
    involve_calc = _to_bool(model_detail.get("involveCalc"))

    model_meta = {
        "code": _to_text(model_detail.get("code") or model_detail.get("modelCode")),
        "modelName": _to_text(model_detail.get("modelName") or model_detail.get("name")),
        "modelDesc": _to_text(model_detail.get("modelDesc") or model_detail.get("description")),
        "modelType": _to_text(model_detail.get("modelType") or model_detail.get("type")),
        "businessModelType": model_type,
        "involveCalc": involve_calc,
        "referenceModelCode": _to_text(model_detail.get("referenceModelCode")),
        "factory": _to_text(model_detail.get("factory")),
        "format": _to_text(model_detail.get("format")),
        "timeGranularity": _to_text(model_detail.get("timeGranularity"))
    }

    result = []
    seen = set()
    for idx, item in enumerate(field_list):
        if not isinstance(item, dict):
            raise ValueError("fieldList[{}] must be an object".format(idx))
        field_name = _to_text(item.get("fieldName") or item.get("name"))
        if field_name in seen:
            continue
        seen.add(field_name)
        result.append(_build_model_field_item(item, idx, model_type, involve_calc))

    return model_meta, result


def _clean_source_fields(source_fields):
    if source_fields is None or not isinstance(source_fields, list):
        raise ValueError("sourceFields must be a list")

    result = []
    seen = set()
    for idx, item in enumerate(source_fields):
        if not isinstance(item, dict):
            raise ValueError("sourceFields[{}] must be an object".format(idx))

        field_key = _to_text(item.get("fieldKey"))
        field_name = _to_text(item.get("fieldName"))
        source_table = _to_text(item.get("sourceTable"))

        if not field_key:
            raise ValueError("sourceFields[{}].fieldKey is required".format(idx))
        if not field_name:
            raise ValueError("sourceFields[{}].fieldName is required".format(idx))
        if not source_table:
            raise ValueError("sourceFields[{}].sourceTable is required".format(idx))
        if field_key in seen:
            continue
        seen.add(field_key)

        result.append({
            "fieldKey": field_key,
            "fieldName": field_name,
            "sourceTable": source_table,
            "fieldType": _to_text(item.get("fieldType")).lower(),
            "sampleValue": _to_text(item.get("sampleValue"))
        })

    return result


def _clean_source_data(source_data):
    if source_data is None:
        return {}
    if not isinstance(source_data, dict):
        raise ValueError("sourceData must be an object")

    result = {}
    for table, rows in source_data.items():
        table_name = _to_text(table)
        if not table_name:
            continue
        if not isinstance(rows, list):
            continue
        clean_rows = [row for row in rows if isinstance(row, dict)]
        if clean_rows:
            result[table_name] = clean_rows
    return result


def _clean_mappings(mappings, model_fields, source_fields):
    if mappings is None or not isinstance(mappings, dict):
        raise ValueError("mappings must be an object")

    valid_targets = {item["fieldName"] for item in model_fields}
    valid_sources = {item["fieldKey"] for item in source_fields}
    cleaned = {}

    for target_name, source_keys in mappings.items():
        target = _to_text(target_name)
        if target not in valid_targets:
            continue

        if isinstance(source_keys, str):
            source_keys = [source_keys]
        if not isinstance(source_keys, list):
            continue

        valid = _unique_keep_order([_to_text(key) for key in source_keys if _to_text(key) in valid_sources])
        if valid:
            cleaned[target] = valid

    return cleaned


def _clean_dsl_definitions(dsl_definitions):
    if dsl_definitions is None or not isinstance(dsl_definitions, dict):
        raise ValueError("dslDefinitions must be an object")

    required = {"filter", "transform", "sort"}
    if not required.issubset(set(dsl_definitions.keys())):
        raise ValueError("dslDefinitions must contain filter/transform/sort")

    transform = dsl_definitions.get("transform")
    if not isinstance(transform, dict):
        raise ValueError("dslDefinitions.transform must be an object")

    has_types = isinstance(transform.get("types"), list)
    has_operator_defs = isinstance(transform.get("operators"), list)
    if not (has_types or has_operator_defs):
        raise ValueError("dslDefinitions.transform must contain types or operators")

    return dsl_definitions


def _get_allowed_transform_types(dsl_definitions):
    transform = dsl_definitions.get("transform")
    if not isinstance(transform, dict):
        return set(TRANSFORM_TYPES)

    types = []
    if isinstance(transform.get("types"), list):
        types.extend([_to_text(item).lower() for item in transform.get("types")])

    if isinstance(transform.get("operators"), list):
        for item in transform.get("operators"):
            if isinstance(item, dict):
                operator_type = _to_text(item.get("type")).lower()
                if operator_type:
                    types.append(operator_type)

    allowed = {item for item in types if item in TRANSFORM_TYPES}
    return allowed if allowed else set(TRANSFORM_TYPES)


def _resolve_mapping_batch_size():
    return DEFAULT_MAPPING_BATCH_SIZE



def _chunk_by_size(items, batch_size):
    if batch_size <= 0:
        batch_size = DEFAULT_MAPPING_BATCH_SIZE
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def _build_batch_context(batch_targets, model_fields, source_fields, source_data, mappings):
    target_set = set(batch_targets)

    batch_model_fields = [item for item in model_fields if item.get("fieldName") in target_set]

    batch_mappings = {}
    used_source_keys = set()
    for target in batch_targets:
        source_keys = mappings.get(target, [])
        if not source_keys:
            continue
        batch_mappings[target] = source_keys
        for source_key in source_keys:
            used_source_keys.add(source_key)

    batch_source_fields = [item for item in source_fields if item.get("fieldKey") in used_source_keys]

    table_columns = {}
    for source_key in used_source_keys:
        table_name, column_name = _parse_source_key(source_key)
        if not table_name or not column_name:
            continue
        if table_name not in table_columns:
            table_columns[table_name] = set()
        table_columns[table_name].add(column_name)

    batch_source_data = {}
    for table_name, columns in table_columns.items():
        rows = source_data.get(table_name, [])
        if not rows:
            continue

        compact_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            compact_row = {}
            for column in columns:
                if column in row:
                    compact_row[column] = row.get(column)
            if compact_row:
                compact_rows.append(compact_row)

        if compact_rows:
            batch_source_data[table_name] = compact_rows

    return {
        "modelFields": batch_model_fields,
        "sourceFields": batch_source_fields,
        "sourceData": batch_source_data,
        "mappings": batch_mappings
    }


def _is_numeric_text(value):
    text = _to_text(value)
    if not text:
        return False
    normalized = text.replace(",", "").replace("%", "")
    try:
        float(normalized)
        return True
    except (TypeError, ValueError):
        return False


def _is_time_field(field):
    if field.get("businessType") == "time":
        return True

    target = "{} {} {} {}".format(
        field.get("fieldName", ""),
        field.get("businessDesc", ""),
        field.get("targetFormat", ""),
        field.get("fieldType", "")
    ).lower()
    return any(keyword in target for keyword in TIME_KEYWORDS)


def _parse_source_key(field_key):
    text = _to_text(field_key)
    parts = text.split(".", 1)
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _collect_values_by_target(field_name, mappings, source_data):
    values = []
    for key in mappings.get(field_name, []):
        table_name, column_name = _parse_source_key(key)
        if not table_name or not column_name:
            continue
        rows = source_data.get(table_name, [])
        for row in rows:
            if column_name in row:
                values.append(row.get(column_name))
    return values


def _detect_origin_type_from_mapped_source(field_name, mappings, source_data):
    mapped_keys = mappings.get(field_name, [])
    if len(mapped_keys) != 1:
        return ""

    values = _collect_values_by_target(field_name, mappings, source_data)
    for value in values:
        origin = _infer_origin_type_from_value(value)
        if origin:
            return origin
    return ""


def _detect_non_numeric_tokens(values):
    found = []
    for value in values:
        text = _to_text(value)
        if not text:
            continue
        if _is_numeric_text(text):
            continue
        lowered = text.lower()
        if lowered in NON_NUMERIC_TOKEN_BLACKLIST:
            found.append(text)
            continue
        if re.match(r"^[a-zA-Z_\-]+$", text):
            found.append(text)
    return _unique_keep_order(found)[:6]



def _infer_origin_type_from_value(value):
    text = _to_text(value)
    if not text:
        return ""

    patterns = [
        (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
        (r"^\d{4}/\d{2}/\d{2}$", "YYYY/MM/DD"),
        (r"^\d{4}\.\d{2}\.\d{2}$", "YYYY.MM.DD"),
        (r"^\d{4}年\d{1,2}月\d{1,2}日$", "YYYY年MM月DD日"),
        (r"^\d{4}-\d{2}$", "YYYY-MM"),
        (r"^\d{4}/\d{2}$", "YYYY/MM"),
        (r"^\d{2}:\d{2}:\d{2}$", "hh:mm:ss"),
        (r"^\d{2}:\d{2}$", "hh:mm"),
        (r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$", "YYYY-MM-DD hh:mm:ss"),
        (r"^\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}$", "YYYY/MM/DD hh:mm:ss")
    ]

    for pattern, template in patterns:
        if re.match(pattern, text):
            return template

    return ""


def _normalize_time_template(template):
    return _to_text(template).replace(" ", "").upper()


def _is_same_time_template(origin_template, target_template):
    if not origin_template or not target_template:
        return False
    return _normalize_time_template(origin_template) == _normalize_time_template(target_template)


def _is_valid_origin_type_template(origin_type):
    text = _to_text(origin_type)
    if not text:
        return False
    if not TIME_TOKEN_PATTERN.search(text):
        return False
    stripped = TIME_TOKEN_PATTERN.sub("", text)
    ascii_only = stripped.replace("T", "")
    return not re.search(r"[A-Za-z]", ascii_only)


def _sanitize_filter(raw_filter):
    if not isinstance(raw_filter, dict):
        return None

    mode = _to_text(raw_filter.get("mode") or "simple").lower()
    if mode == "compound":
        logic = "OR" if _to_text(raw_filter.get("logic") or "AND").upper() == "OR" else "AND"
        conditions = []
        for cond in raw_filter.get("conditions", []):
            if not isinstance(cond, dict):
                continue
            operator = _to_text(cond.get("operator")).lower()
            if operator not in FILTER_OPERATORS:
                continue
            conditions.append({"operator": operator, "value": cond.get("value", "")})
        if not conditions:
            return None
        return {
            "mode": "compound",
            "logic": logic,
            "conditions": conditions,
            "operator": "",
            "value": "",
            "formula": ""
        }

    operator = _to_text(raw_filter.get("operator")).lower()
    if operator not in FILTER_OPERATORS:
        return None

    return {
        "mode": "simple",
        "operator": operator,
        "value": raw_filter.get("value", ""),
        "formula": "",
        "conditions": [],
        "logic": "AND"
    }


def _sanitize_transform_step(raw_step, allow_origin_type=False, target_format="", allowed_transform_types=None):
    if not isinstance(raw_step, dict):
        return None

    allowed = allowed_transform_types or set(TRANSFORM_TYPES)
    allowed_with_alias = set(allowed)
    if "format_datetime" in allowed:
        allowed_with_alias = allowed_with_alias.union(TIME_TRANSFORM_INPUT_TYPES)

    transform_type = _normalize_transform_type(raw_step.get("type"))
    if transform_type not in allowed_with_alias:
        return None

    item = {
        "type": transform_type,
        "delimiter": raw_step.get("delimiter", ""),
        "fixedValue": raw_step.get("fixedValue", ""),
        "search": raw_step.get("search", ""),
        "replace": raw_step.get("replace", ""),
        "formula": ""
    }

    if transform_type == "format_datetime":
        origin_type = _to_text(raw_step.get("originType"))
        if not (allow_origin_type and _is_valid_origin_type_template(origin_type)):
            return None
        item["originType"] = origin_type

    return item


def _sanitize_transform(field_name, raw_transform, mappings, model_field_map, allowed_transform_types):
    if not isinstance(raw_transform, dict):
        return None

    source_count = len(mappings.get(field_name, []))
    model_field = model_field_map.get(field_name, {})
    allow_origin_type = source_count == 1 and _is_time_field(model_field)
    target_format = model_field.get("targetFormat")

    chain = raw_transform.get("chain")
    if isinstance(chain, list) and chain:
        sanitized_chain = [
            item for item in (
                _sanitize_transform_step(step, allow_origin_type, target_format, allowed_transform_types)
                for step in chain
            ) if item
        ]
        if not sanitized_chain:
            return None
        return {"rules": [], "chain": sanitized_chain}

    rules = raw_transform.get("rules")
    if isinstance(rules, list) and rules:
        sanitized_rules = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            operator = _to_text(rule.get("operator")).lower()
            if operator not in FILTER_OPERATORS:
                continue
            step = _sanitize_transform_step(rule, allow_origin_type, target_format, allowed_transform_types)
            if not step:
                continue
            sanitized_rules.append({
                "operator": operator,
                "value": rule.get("value", ""),
                "type": step["type"],
                "delimiter": step.get("delimiter", ""),
                "fixedValue": step.get("fixedValue", ""),
                "search": step.get("search", ""),
                "replace": step.get("replace", ""),
                "formula": "",
                "originType": step.get("originType", "")
            })
        if not sanitized_rules:
            return None
        return {"rules": sanitized_rules, "chain": []}

    single = _sanitize_transform_step(raw_transform, allow_origin_type, target_format, allowed_transform_types)
    if not single:
        return None

    single["rules"] = []
    single["chain"] = []
    return single


def _sanitize_sort(raw_sort):
    if not isinstance(raw_sort, dict):
        return None
    order = _to_text(raw_sort.get("order") or raw_sort.get("direction") or "asc").lower()
    if order not in {"asc", "desc"}:
        return None
    return {"order": order}


def _sanitize_suggestions(raw_payload, model_fields, mappings, dsl_definitions):
    source = raw_payload.get("suggestions", raw_payload)
    if not isinstance(source, dict):
        return {}

    model_field_map = {item["fieldName"]: item for item in model_fields}
    ordered_fields = [item["fieldName"] for item in model_fields]
    allowed_transform_types = _get_allowed_transform_types(dsl_definitions)
    cleaned = {}

    for field_name in ordered_fields:
        raw_item = source.get(field_name)
        if not isinstance(raw_item, dict):
            continue

        operations = raw_item.get("operations") if isinstance(raw_item.get("operations"), dict) else {}
        if not operations and isinstance(raw_item.get("config"), dict):
            operations = {"transform": raw_item.get("config")}

        clean_ops = {}
        clean_filter = _sanitize_filter(operations.get("filter"))
        clean_transform = _sanitize_transform(
            field_name, operations.get("transform"), mappings, model_field_map, allowed_transform_types
        )
        clean_sort = _sanitize_sort(operations.get("sort"))

        if clean_filter:
            clean_ops["filter"] = clean_filter
        if clean_transform:
            clean_ops["transform"] = clean_transform
        if clean_sort:
            clean_ops["sort"] = clean_sort

        if not clean_ops:
            continue

        cleaned[field_name] = {
            "needAttention": raw_item.get("needAttention", True) is not False,
            "hint": _to_text(raw_item.get("hint") or raw_item.get("suggestion") or "建议检查该列处理配置"),
            "operations": clean_ops
        }

    return cleaned


def _build_counter_kpi_fallback(field, field_name, mapped_keys, mappings, source_data, allowed_transform_types):
    """构建counter/kpi类型字段的数值清理fallback"""
    model_type = _to_text(field.get("modelType")).lower()
    business_type = _to_text(field.get("businessType")).lower()
    if not (model_type in {"counter", "kpi"} or business_type == "metric"):
        return None

    values = _collect_values_by_target(field_name, mappings, source_data)
    abnormal_tokens = _detect_non_numeric_tokens(values)
    if not abnormal_tokens:
        return None

    replacement = "0" if field.get("involveCalc") else ""
    rules = [
        {"operator": "equals", "value": token, "type": "set_value", "fixedValue": replacement}
        for token in abnormal_tokens
    ]
    return {
        "needAttention": True,
        "hint": "检测到非数值异常值，建议先替换后再计算",
        "operations": {"transform": {"rules": rules}}
    }


def _build_time_fallback(field, field_name, mapped_keys, mappings, source_data, allowed_transform_types):
    """构建时间字段的格式化fallback"""
    target_template = _to_text(field.get("targetFormat"))
    if not target_template:
        return None

    origin_type = _detect_origin_type_from_mapped_source(field_name, mappings, source_data)
    if not origin_type:
        return None
    if _is_same_time_template(origin_type, target_template):
        return None

    transform = {
        "type": "format_datetime",
        "originType": origin_type if len(mapped_keys) == 1 else ""
    }
    return {
        "needAttention": True,
        "hint": "检测到源时间格式与目标格式不一致，建议使用格式化时间",
        "operations": {"transform": transform}
    }


def _build_fallback_suggestions(model_fields, mappings, source_data, dsl_definitions):
    """构建规则建议（仅时间格式化规则）"""
    allowed_transform_types = _get_allowed_transform_types(dsl_definitions)
    result = {}

    for field in model_fields:
        field_name = field["fieldName"]
        mapped_keys = mappings.get(field_name, [])
        if not mapped_keys:
            continue

        # Time formatting fallback
        if "format_datetime" in allowed_transform_types and _is_time_field(field):
            suggestion = _build_time_fallback(
                field, field_name, mapped_keys, mappings, source_data, allowed_transform_types
            )
            if suggestion:
                result[field_name] = suggestion

    return result


def _execute_llm_batches(mapped_targets, model_fields, source_fields, source_data, mappings, dsl_definitions, model_meta, chat_fn, max_retries=3):
    """执行LLM批次调用，支持重试机制

    Args:
        max_retries: 最大重试次数，默认3次
    """
    batch_size = _resolve_mapping_batch_size()
    batches = list(_chunk_by_size(mapped_targets, batch_size))
    logger.info(
        "generate_process_suggestions batching, target_count:%s, batch_size:%s, batch_count:%s, max_retries:%s",
        len(mapped_targets), batch_size, len(batches), max_retries
    )

    llm_suggestions = {}
    llm_errors = []

    for batch_index, batch_targets in enumerate(batches, start=1):
        batch_context = _build_batch_context(batch_targets, model_fields, source_fields, source_data, mappings)
        batch_model_fields = batch_context["modelFields"]
        batch_source_fields = batch_context["sourceFields"]
        batch_source_data = batch_context["sourceData"]
        batch_mappings = batch_context["mappings"]

        if not batch_model_fields or not batch_source_fields or not batch_mappings:
            logger.warning("generate_process_suggestions skip batch %s due to empty context", batch_index)
            continue

        payload = {
            "task": "generate_process_suggestions",
            "modelDetail": model_meta,
            "modelFields": batch_model_fields,
            "sourceFields": batch_source_fields,
            "sourceData": batch_source_data,
            "mappings": batch_mappings,
            "dslDefinitions": dsl_definitions
        }

        batch_success = False
        last_error = None
        for retry in range(max_retries):
            try:
                messages = build_messages(payload)
                llm_text = chat_fn(messages)
                llm_payload = extract_json_dict(llm_text)
                batch_suggestions = _sanitize_suggestions(llm_payload, batch_model_fields, batch_mappings, dsl_definitions)
                llm_suggestions.update(batch_suggestions)
                batch_success = True
                if retry > 0:
                    logger.info("generate_process_suggestions batch %s succeeded after %d retries", batch_index, retry)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if retry < max_retries - 1:
                    logger.warning("generate_process_suggestions batch %s failed (retry %d/%d): %s",
                                   batch_index, retry + 1, max_retries, exc)
                else:
                    logger.error("generate_process_suggestions batch %s failed after %d retries: %s",
                                 batch_index, max_retries, exc)

        if not batch_success:
            llm_errors.append("batch{}_{}".format(batch_index, last_error.__class__.__name__ if last_error else "Unknown"))

    return llm_suggestions, llm_errors


def _filter_time_space_only(model_fields, source_fields, source_data, mappings):
    """过滤数据，只保留fieldBusinessType为time或space的字段

    这样可以大大缩减需要LLM处理的数据量。
    """
    # 筛选fieldBusinessType为time或space的字段
    allowed_business_types = {"time", "space"}
    target_fields = [
        f for f in model_fields
        if _to_text(f.get("businessType")).lower() in allowed_business_types
    ]

    if not target_fields:
        logger.warning("filter_time_space_only: no time/space fields found, keeping all fields")
        return model_fields, source_fields, source_data, mappings

    target_field_names = {f["fieldName"] for f in target_fields}

    # 过滤mappings，只保留目标字段中的映射
    filtered_mappings = {
        k: v for k, v in mappings.items() if k in target_field_names
    }

    # 收集被使用的source字段key
    used_source_keys = set()
    for source_keys in filtered_mappings.values():
        used_source_keys.update(source_keys)

    # 过滤source_fields
    filtered_source_fields = [
        s for s in source_fields if s.get("fieldKey") in used_source_keys
    ]

    # 过滤source_data，只保留被使用的列
    filtered_source_data = {}
    for table_name, rows in source_data.items():
        if not rows:
            continue
        # 收集该表中被使用的列
        used_columns = set()
        for source_field in filtered_source_fields:
            if source_field.get("sourceTable") == table_name:
                key = source_field.get("fieldKey", "")
                _, col_name = _parse_source_key(key)
                if col_name:
                    used_columns.add(col_name)

        if used_columns:
            filtered_rows = []
            for row in rows:
                if isinstance(row, dict):
                    filtered_row = {col: row.get(col) for col in used_columns if col in row}
                    if filtered_row:
                        filtered_rows.append(filtered_row)
            if filtered_rows:
                filtered_source_data[table_name] = filtered_rows

    logger.info(
        "filter_time_space_only: original %d fields -> filtered %d fields, "
        "original %d source_fields -> filtered %d source_fields",
        len(model_fields), len(target_fields),
        len(source_fields), len(filtered_source_fields)
    )

    return target_fields, filtered_source_fields, filtered_source_data, filtered_mappings


def _merge_suggestions(llm_suggestions, fallback_suggestions, model_fields, llm_errors):
    """合并LLM建议和fallback建议"""
    llm_error = "llm_failed:{}".format("|".join(_unique_keep_order(llm_errors))) if llm_errors else ""

    if llm_suggestions:
        merged = {}
        fallback_applied = False
        for item in model_fields:
            name = item["fieldName"]
            if name in llm_suggestions:
                merged[name] = llm_suggestions[name]
            elif name in fallback_suggestions:
                merged[name] = fallback_suggestions[name]
                fallback_applied = True

        fallback_reason = ""
        if fallback_applied:
            fallback_reason = "fallback_applied:partial"
            if llm_error:
                fallback_reason = fallback_reason + ";" + llm_error

        return {
            "suggestions": merged,
            "fallbackApplied": fallback_applied,
            "fallbackReason": fallback_reason
        }

    if fallback_suggestions:
        return {
            "suggestions": fallback_suggestions,
            "fallbackApplied": True,
            "fallbackReason": llm_error or "fallback_applied:llm_empty_or_invalid"
        }

    return {
        "suggestions": {},
        "fallbackApplied": bool(llm_error),
        "fallbackReason": llm_error
    }


def generate_process_suggestions(model_detail, source_fields, source_data, mappings, dsl_definitions, llm_chat_fn=None):
    """Return dict with suggestions and fallback metadata."""
    model_meta, model_fields = _clean_model_detail(model_detail)
    source_fields = _clean_source_fields(source_fields)
    source_data = _clean_source_data(source_data)
    dsl_definitions = _clean_dsl_definitions(dsl_definitions)
    mappings = _clean_mappings(mappings, model_fields, source_fields)

    if not model_fields or not source_fields:
        return {"suggestions": {}, "fallbackApplied": False, "fallbackReason": ""}

    # 预处理：只保留time和space类型的字段，减少LLM处理的数据量
    model_fields, source_fields, source_data, mappings = _filter_time_space_only(
        model_fields, source_fields, source_data, mappings
    )

    if not model_fields or not source_fields:
        return {"suggestions": {}, "fallbackApplied": False, "fallbackReason": ""}

    fallback_suggestions = _build_fallback_suggestions(model_fields, mappings, source_data, dsl_definitions)

    mapped_targets = [item["fieldName"] for item in model_fields if mappings.get(item["fieldName"])]
    if not mapped_targets:
        return {"suggestions": {}, "fallbackApplied": False, "fallbackReason": ""}

    chat_fn = llm_chat_fn or chat_completion
    llm_suggestions, llm_errors = _execute_llm_batches(
        mapped_targets, model_fields, source_fields, source_data, mappings, dsl_definitions, model_meta, chat_fn
    )

    return _merge_suggestions(llm_suggestions, fallback_suggestions, model_fields, llm_errors)
