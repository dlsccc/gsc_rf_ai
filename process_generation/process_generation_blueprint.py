#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.

from flask import Blueprint, request
from psf.psflogging.psf_log import PsfLog

from constants import response_constants

from utils.log_utils import LogUtils
from service.data_smart import data_process_generation_service

logger = LogUtils.get_logger(__name__)
bp = Blueprint('process_generation', __name__)
psf_log = PsfLog.psf_log


@bp.route("/v1/dataSmart/generateProcessConfig", methods=['POST'])
@psf_log(object_type="data", operation_name="generateProcessConfig", module_name="data_smart",
         param_name_list=["code", "modelName", "businessModelType", "involveCalc", "fieldList",
                          "sourceFields", "sourceData", "mappings", "dslDefinitions"])
def generate_process_config():
    """自动配置生成接口：调用大模型完成字段数据处理配置生成"""
    try:
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}

        logger.info(
            "generate_process_config enter, fieldList size: %s, sourceFields size: %s",
            len(data.get("fieldList", data.get("modelFields", []))),
            len(data.get("sourceFields", []))
        )

        result = data_process_generation_service.execute_generate_process_config(data)
        logger.info(
            "generate_process_config success, suggestion size: %s, fallbackApplied: %s",
            len(result.get("suggestions", {})),
            result.get("fallbackApplied", False)
        )
        return response_constants.update_success_result_data(result)
    except ValueError as exc:
        logger.error("generate_process_config validation error: %s", exc)
        return response_constants.FAILED
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_process_config error: %s", exc)
        return response_constants.FAILED


