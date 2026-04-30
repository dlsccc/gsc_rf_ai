#!/usr/bin/env python
# coding=utf-8
#  Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.


import json

from service.llm_service.llm_factory import LlmFactory
from utils.log_utils import LogUtils

logger = LogUtils.get_logger('data_smart_process_call_llm')

DEFAULT_MODEL = "qwen3.5"

SYSTEM_PROMPT = """
你是数据处理规则生成助手。你需要基于输入数据，为目标字段输出“可直接应用”的处理建议。
输入数据及任务解释：
1) fieldList是定义的项目模型，包含了各个字段的类型、格式和业务类型等定义
2) sourceFields是原始数据的字段解释，sourceData是真实上传的原始数据
3) mappings是原始数据字段到项目模型的映射关系
4) 这里的任务是将原始数据处理成项目模型中规定的数据的样子，生成需要的算子配置

必须遵守：
1) 只能使用输入中的 modelDetail.fieldList（以及归一化后的 modelFields）/sourceFields/mappings/sourceData，不得臆造字段。
2) 只能使用给定 dslDefinitions 中的算子，且 transform 参数必须遵循 dslDefinitions.transform.operators[*] 的 params/required。
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
7) 只需要处理时间格式转换：
     - 只有当源数据的时间格式与目标字段的 dataFormat 不一致时，才建议使用 format_datetime
     - 禁止生成以下转换：
       * 数值类型转字符串（如 float -> string）
       * 字符串转数值（如 string -> int/float）
       * 不要用set_value进行数据格式的转换！！！
8) 不确定时可不返回该字段建议，不要猜测。
9) 若某算子缺少必填参数，不要输出该算子。

算子说明（以 dslDefinitions 定义为准，下面仅作参考）：
A. filter
- simple: {"mode":"simple","operator":"equals|not_equals|contains|is_empty|is_not_empty|greater_than|less_than","value":"..."}
- compound: {"mode":"compound","logic":"AND|OR","conditions":[{"operator":"...","value":"..."}]}

B. transform（只允许以下 type）
- format_datetime: 只输出 originType（例如 YYYY-MM-DD / YYYY/MM/DD / hh:mm:ss）
- custom originType is allowed when the source time format is not covered by preset options; for example, source sample `2026/3/2` should infer `YYYY/MM/DD`.
- calc_week
- calc_weekday
- set_value（fixedValue）
- concat（delimiter）
- replace（search/replace）

C. sort
- {"order":"asc|desc"}

需要严格遵守的业务规则：
- 当字段业务类型为time时，先根据对应的原始数据推断字段的原始时间格式originType，再看是否与模型中要求的格式相同，如果不同，则需要调用format_datetime算子
- 字段类型如果不一致不需要进行转换修改，比如原始数据是int，目标字段要求是string，不需要转换格式
- 不要编造处理方法

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
