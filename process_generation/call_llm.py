#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
"""LLM client for process suggestion generation."""

import json

from service.llm_service.llm_factory import LlmFactory
from utils.log_utils import LogUtils

logger = LogUtils.get_logger('data_smart_process_call_llm')

DEFAULT_MODEL = "qwen-s-pro"

SYSTEM_PROMPT = """
你是数据处理规则生成助手。你需要基于输入数据，为目标字段输出“可直接应用”的处理建议。

必须遵守：
1) 只能使用输入中的 modelDetail.fieldList（以及归一化后的 modelFields）/sourceFields/mappings/sourceData，不得臆造字段。
2) 只能使用给定 dslDefinitions 中的算子。
3) 严禁输出 formula 或任何未定义算子。
4) 输出必须是严格 JSON，且只输出 JSON，不要解释。
5) 建议按列输出，格式：
{
  "suggestions": {
    "<targetFieldName>": {
      "needAttention": true,
      "hint": "...",
      "operations": {
        "filter": {...},
        "transform": {...},
        "sort": {...}
      }
    }
  }
}
6) 时间转换规则：
   - 先推断源字段 originType（源格式模板）和目标字段 targetFormat（模型字段格式）。
   - 当 originType 与 targetFormat 不一致时，才建议使用 format_datetime。
   - format_datetime 时，必须输出 originType（源格式模板字符串：YYYY/MM/DD/hh/mm/ss + 原分隔符）；非时间字段或多源映射字段不要输出 originType。
7) 不确定时可不返回该字段建议，不要猜测。

算子说明：
A. filter
- simple: {"mode":"simple","operator":"equals|not_equals|contains|is_empty|is_not_empty|greater_than|less_than","value":"..."}
- compound: {"mode":"compound","logic":"AND|OR","conditions":[{"operator":"...","value":"..."}]}

B. transform（只允许以下 type）
- format_datetime: 只输出 originType（例如 YYYY-MM-DD / YYYY/MM/DD / hh:mm:ss）
- calc_week
- calc_weekday
- set_value（fixedValue）
- concat（delimiter）
- replace（search/replace）

C. sort
- {"order":"asc|desc"}

业务规则：
- 当 modelType 为 counter/kpi 且 involveCalc=true，若发现非数值异常值，建议替换为 "0"。
- 当 modelType 为 counter/kpi 且 involveCalc=false，若发现非数值异常值，建议替换为空字符串 ""。
""".strip()


def chat_completion(messages, model=DEFAULT_MODEL):
    """Call LLM and return plain text response."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append("[System]\n{}".format(content))
        else:
            prompt_parts.append(str(content))

    prompt = "\n".join(prompt_parts)
    model_use = LlmFactory.get_model(model_name=model)
    result = model_use.send_request(prompt=prompt)
    logger.info("process_generation chat_completion done")
    return result if isinstance(result, str) else str(result)


def build_messages(payload):
    """Build chat messages for process suggestion task."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]


def extract_json_dict(text):
    """Extract JSON object from raw LLM response text."""
    raw = str(text or "").strip()
    if not raw:
        raise RuntimeError("llm response is empty")

    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
    else:
        candidate = raw

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError("llm response does not contain valid json") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("llm response json must be object")
    return payload






