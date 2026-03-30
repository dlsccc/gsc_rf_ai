#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
"""Service layer for auto field mapping."""

from .mapper  import auto_map_fields as run_auto_map_fields


def _ensure_dict(data):
    return data if isinstance(data, dict) else {}


def _build_response(result):
    response = {
        "mappings": result.get("mappings", {})
    }
    if result.get("fallbackApplied"):
        response["fallbackApplied"] = True
        reason = result.get("fallbackReason")
        if reason:
            response["fallbackReason"] = reason
    return response


def execute_auto_map_fields(data):
    payload = _ensure_dict(data)
    model_fields = payload.get("modelFields", [])
    source_fields = payload.get("sourceFields", [])

    result = run_auto_map_fields(model_fields=model_fields, source_fields=source_fields)
    return _build_response(result)

