#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi.testclient import TestClient

import column_mapping.app as app_module
from column_mapping.mapper import AutoMapResult

client = TestClient(app_module.app)


def test_api_returns_mappings_structure(monkeypatch):
    def fake_runner(model_fields, source_fields):
        assert model_fields and source_fields
        return AutoMapResult(mappings={"imsi": ["table_a.imsi"]}, fallback_applied=False)

    monkeypatch.setattr(app_module, "run_auto_map_fields", fake_runner)

    response = client.post(
        "/itsc/lingluoservice/dataSmart/autoMapFields",
        json={
            "modelFields": [{"fieldName": "imsi"}],
            "sourceFields": [{"fieldKey": "table_a.imsi", "fieldName": "imsi", "sourceTable": "table_a"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "1"
    assert body["data"]["mappings"] == {"imsi": ["table_a.imsi"]}


def test_api_marks_fallback_in_msg(monkeypatch):
    def fake_runner(model_fields, source_fields):
        assert model_fields and source_fields
        return AutoMapResult(
            mappings={"location": ["table_a.lng", "table_a.lat"]},
            fallback_applied=True,
            fallback_reason="fallback_applied:partial",
        )

    monkeypatch.setattr(app_module, "run_auto_map_fields", fake_runner)

    response = client.post(
        "/itsc/lingluoservice/dataSmart/autoMapFields",
        json={
            "modelFields": [{"fieldName": "location"}],
            "sourceFields": [
                {"fieldKey": "table_a.lng", "fieldName": "lng", "sourceTable": "table_a"},
                {"fieldKey": "table_a.lat", "fieldName": "lat", "sourceTable": "table_a"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "1"
    assert "fallback_applied" in body["msg"]


def test_api_returns_code_0_for_missing_required_fields():
    response = client.post(
        "/itsc/lingluoservice/dataSmart/autoMapFields",
        json={"modelFields": [{}], "sourceFields": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "0"
    assert "fieldName is required" in body["msg"]
