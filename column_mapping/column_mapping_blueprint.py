#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.

from flask import Blueprint, request
from psf.psflogging.psf_log import PsfLog

from constants import response_constants

from utils.log_utils import LogUtils

from service.data_smart import data_mapping_service

logger = LogUtils.get_logger(__name__)
bp = Blueprint('data_smart', __name__)
psf_log = PsfLog.psf_log


@bp.route("/v1/dataSmart/autoMapFields", methods=['POST'])
@psf_log(object_type="data", operation_name="autoMapFields", module_name="data_smart",
         param_name_list=["modelFields", "sourceFields"])
def auto_map_fields():
    """
    自动字段映射接口：调用大模型完成目标字段与源字段匹配，失败时回退规则匹配
    :param modelFields: 目标模型字段列表
    :param sourceFields: 源字段列表
    :return: 字段映射关系（支持一对多）
    """
    try:
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}

        logger.info(
            f"auto_map_fields enter, modelFields size: {len(data.get('modelFields', []))}, "
            f"sourceFields size: {len(data.get('sourceFields', []))}"
        )

        result = data_mapping_service.execute_auto_map_fields(data)

        logger.info(
            f"auto_map_fields success, mapping size: {len(result.get('mappings', {}))}, "
            f"fallbackApplied: {result.get('fallbackApplied', False)}"
        )
        return response_constants.update_success_result_data(result)
    except ValueError as e:
        logger.error(f"auto_map_fields validation error: {e}")
        return response_constants.FAILED
    except Exception as e:
        logger.error(f"auto_map_fields error: {e}")
        return response_constants.FAILED