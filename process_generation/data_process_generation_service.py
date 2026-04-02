#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
"""Service layer for process suggestion generation."""

import json
from .mapper import generate_process_suggestions as run_generate_process_suggestions


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



def _build_local_test_payload():
    """Build one local test payload in API request format."""
    return {
        "code": "PROJECT_MODEL_001",
        "modelName": "4G???KPI??",
        "modelDesc": "???????",
        "modelType": "business",
        "referenceModelCode": "STD_4G_COUNTER",
        "factory": "HW",
        "format": "FDD",
        "timeGranularity": "hour",
        "businessModelType": "kpi",
        "involveCalc": False,
        "fieldList": [
            {
                "modelCode": "PROJECT_MODEL_001",
                "fieldName": "TIME_MONTH",
                "fieldType": "STRING",
                "fieldDesc": "??",
                "dataFormat": "YYYY-MM",
                "dataExample": "2026-03",
                "fieldBusinessType": "time",
                "isNull": True,
                "seq": 1
            },
            {
                "modelCode": "PROJECT_MODEL_001",
                "fieldName": "KPI_VALUE",
                "fieldType": "FLOAT64",
                "fieldDesc": "???",
                "dataFormat": "FLOAT",
                "dataExample": "12.6",
                "fieldBusinessType": "metric",
                "isNull": True,
                "seq": 2
            }
        ],
        "sourceFields": [
            {
                "fieldKey": "table_a.date",
                "fieldName": "date",
                "sourceTable": "table_a",
                "fieldType": "string",
                "sampleValue": "2026/03/31"
            },
            {
                "fieldKey": "table_a.kpi",
                "fieldName": "kpi",
                "sourceTable": "table_a",
                "fieldType": "string",
                "sampleValue": "NIL"
            }
        ],
        "sourceData": {
            "table_a": [
                {"date": "2026/03/31", "kpi": "12.6"},
                {"date": "2026/03/30", "kpi": "NIL"}
            ]
        },
        "mappings": {
            "TIME_MONTH": ["table_a.date"],
            "KPI_VALUE": ["table_a.kpi"]
        },
        "dslDefinitions": {
            "filter": {
                "modes": ["simple", "compound"],
                "operators": ["equals", "not_equals", "contains", "is_empty", "is_not_empty", "greater_than", "less_than"],
                "logic": ["AND", "OR"]
            },
            "transform": {
                "types": ["format_datetime", "calc_week", "calc_weekday", "set_value", "concat", "replace"],
                "operators": [
                    {"type": "format_datetime", "params": [{"name": "originType", "type": "string", "required": True}], "required": ["originType"]},
                    {"type": "calc_week", "params": [], "required": []},
                    {"type": "calc_weekday", "params": [], "required": []},
                    {"type": "set_value", "params": [{"name": "fixedValue", "type": "string", "required": True}], "required": ["fixedValue"]},
                    {"type": "concat", "params": [{"name": "delimiter", "type": "string", "required": False}], "required": []},
                    {"type": "replace", "params": [{"name": "search", "type": "string", "required": True}, {"name": "replace", "type": "string", "required": True}], "required": ["search", "replace"]}
                ],
                "disallow": ["formula"]
            },
            "sort": {
                "orders": ["asc", "desc"]
            }
        }
    }


if __name__ == '__main__':
    payload = _build_local_test_payload()
    output = execute_generate_process_config(payload)
    print('===== Local Test Input =====')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('===== Local Test Output =====')
    print(json.dumps(output, ensure_ascii=False, indent=2))
