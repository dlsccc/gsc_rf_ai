#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from column_mapping.mapper import auto_map_fields


def test_auto_map_uses_llm_when_json_is_valid():
    model_fields = [
        {"fieldName": "imsi", "fieldType": "string"},
        {"fieldName": "location", "fieldType": "string"},
    ]
    source_fields = [
        {"fieldKey": "table_a.imsi", "fieldName": "imsi", "sourceTable": "table_a"},
        {"fieldKey": "table_a.lng", "fieldName": "lng", "sourceTable": "table_a"},
        {"fieldKey": "table_a.lat", "fieldName": "lat", "sourceTable": "table_a"},
    ]

    def fake_llm(_messages):
        return """
        {
          "mappings": {
            "imsi": ["table_a.imsi"],
            "location": ["table_a.lng", "table_a.lat"]
          }
        }
        """

    result = auto_map_fields(model_fields=model_fields, source_fields=source_fields, llm_chat_fn=fake_llm)

    assert result["fallbackApplied"] is False
    assert result["mappings"] == {
        "imsi": ["table_a.imsi"],
        "location": ["table_a.lng", "table_a.lat"],
    }


def test_auto_map_fallback_when_llm_returns_invalid_json():
    model_fields = [{"fieldName": "user_name", "fieldType": "string"}]
    source_fields = [{"fieldKey": "table_b.UserName", "fieldName": "UserName", "sourceTable": "table_b"}]

    def fake_llm(_messages):
        return "not-json"

    result = auto_map_fields(model_fields=model_fields, source_fields=source_fields, llm_chat_fn=fake_llm)

    assert result["fallbackApplied"] is True
    assert result["mappings"] == {"user_name": ["table_b.UserName"]}
    assert "llm_failed" in result["fallbackReason"]


def test_auto_map_sanitizes_hallucinated_targets_and_sources():
    model_fields = [{"fieldName": "imsi"}, {"fieldName": "user_name"}]
    source_fields = [
        {"fieldKey": "table_a.imsi", "fieldName": "imsi", "sourceTable": "table_a"},
        {"fieldKey": "table_b.user_name", "fieldName": "user_name", "sourceTable": "table_b"},
    ]

    def fake_llm(_messages):
        return """
        {
          "mappings": {
            "imsi": ["table_a.imsi", "table_a.hallucinated"],
            "fake_target": ["table_b.user_name"],
            "user_name": ["table_b.user_name"]
          }
        }
        """

    result = auto_map_fields(model_fields=model_fields, source_fields=source_fields, llm_chat_fn=fake_llm)

    assert result["mappings"] == {"imsi": ["table_a.imsi"], "user_name": ["table_b.user_name"]}


def test_auto_map_location_uses_lon_lat_rule():
    model_fields = [{"fieldName": "location"}]
    source_fields = [
        {"fieldKey": "table_a.lng", "fieldName": "lng", "sourceTable": "table_a"},
        {"fieldKey": "table_a.lat", "fieldName": "lat", "sourceTable": "table_a"},
    ]

    def fake_llm(_messages):
        return '{"mappings":{}}'

    result = auto_map_fields(model_fields=model_fields, source_fields=source_fields, llm_chat_fn=fake_llm)

    assert result["fallbackApplied"] is True
    assert result["mappings"] == {"location": ["table_a.lng", "table_a.lat"]}


def test_auto_map_same_name_uses_field_key_for_duplicate_names():
    model_fields = [{"fieldName": "user_name"}]
    source_fields = [
        {"fieldKey": "table_a.user_name", "fieldName": "user_name", "sourceTable": "table_a"},
        {"fieldKey": "table_b.user_name", "fieldName": "user_name", "sourceTable": "table_b"},
    ]

    def fake_llm(_messages):
        return '{"mappings":{"user_name":["table_b.user_name"]}}'

    result = auto_map_fields(model_fields=model_fields, source_fields=source_fields, llm_chat_fn=fake_llm)

    assert result["mappings"] == {"user_name": ["table_b.user_name"]}
