#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.


from .process_mapper import generate_process_suggestions as run_generate_process_suggestions


def _ensure_dict(data):
    return data if isinstance(data, dict) else {}


def execute_generate_process_config(data):
    payload = _ensure_dict(data)

    result = run_generate_process_suggestions(
        model_detail=payload,
        source_fields=payload.get("sourceFields", []),
        source_data=payload.get("sourceData", {}),
        mappings=payload.get("mappings", {}),
        dsl_definitions=payload.get("dslDefinitions", {})
    )

    response = {
        "suggestions": result.get("suggestions", {})
    }
    if result.get("fallbackApplied"):
        response["fallbackApplied"] = True
        response["fallbackReason"] = result.get("fallbackReason", "")

    return response
