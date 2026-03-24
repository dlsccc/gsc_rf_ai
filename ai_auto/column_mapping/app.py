#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI app for automatic field mapping."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

try:
    from .mapper import auto_map_fields as run_auto_map_fields
except ImportError:  # pragma: no cover
    from mapper import auto_map_fields as run_auto_map_fields

OK_MSG = "smartdesigner.framework.common.ok"

app = FastAPI(title="Column Mapping Service", version="1.0.0")


class ModelField(BaseModel):
    fieldName: str | None = None
    fieldType: str | None = None
    fieldDesc: str | None = None
    model_config = ConfigDict(extra="ignore")


class SourceField(BaseModel):
    fieldKey: str | None = None
    fieldName: str | None = None
    sourceTable: str | None = None
    fieldType: str | None = None
    sampleValue: str | None = None
    model_config = ConfigDict(extra="ignore")


class AutoMapRequest(BaseModel):
    modelFields: list[ModelField] | None = None
    sourceFields: list[SourceField] | None = None
    model_config = ConfigDict(extra="ignore")


def _success_response(mappings: dict[str, list[str]], msg: str = OK_MSG) -> dict[str, Any]:
    return {
        "code": "1",
        "msg": msg,
        "data": {"mappings": mappings},
        "msgParams": [],
    }


def _error_response(message: str) -> dict[str, Any]:
    return {
        "code": "0",
        "msg": message,
        "data": {"mappings": {}},
        "msgParams": [],
    }


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_request, _exc: RequestValidationError):  # noqa: ANN001
    return JSONResponse(status_code=200, content=_error_response("invalid request body"))


@app.post("/itsc/lingluoservice/dataSmart/autoMapFields")
def auto_map_fields_api(payload: AutoMapRequest):
    try:
        model_fields = [item.model_dump(exclude_none=True) for item in (payload.modelFields or [])]
        source_fields = [item.model_dump(exclude_none=True) for item in (payload.sourceFields or [])]
        result = run_auto_map_fields(model_fields=model_fields, source_fields=source_fields)

        message = OK_MSG
        if result.fallback_applied:
            suffix = result.fallback_reason or "fallback_applied"
            message = f"{message} {suffix}"
        return _success_response(result.mappings, message)
    except ValueError as exc:
        return _error_response(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(f"auto map failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

