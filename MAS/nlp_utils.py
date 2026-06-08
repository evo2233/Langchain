import os
import logging
import re
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class ExtractOptionOutput(BaseModel):
    option: Literal["A", "B", "C", "D", "E", "F"] = Field(
        description="The extracted final option letter."
    )


def _build_default_llm():
    api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8086/v1")
    api_key = os.getenv("VLLM_API_KEY", "vllm")
    return ChatOpenAI(
        model="/model",
        api_key=api_key,
        base_url=api_base,
        max_tokens=32,
        temperature=0.0,
    )


def extract_option(answer_text: str, llm: Optional[ChatOpenAI] = None) -> str:
    """
    使用 LLM 从包含解释的答案文本中抽取单个大写选项字母（如 A/B/C/D）。
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("answer_text must be a non-empty string.")
    
    normalized = answer_text.strip()

    # Fast deterministic path: prefer explicit option patterns from text.
    pattern_order = [
        r"(?i)final\s*answer\s*[:：]\s*\(?\s*([A-F])\s*\)?",
        r"(?i)answer\s*[:：]\s*\(?\s*([A-F])\s*\)?",
        r"(?i)option\s*[:：]\s*\(?\s*([A-F])\s*\)?",
        r"(?i)^\s*\(?\s*([A-F])\s*\)?\s*$",
    ]
    for pattern in pattern_order:
        matches = re.findall(pattern, normalized)
        if matches:
            return matches[-1].upper()

    # Fallback: if isolated option letters appear, use the last one.
    isolated = re.findall(r"(?<![A-Z])([A-F])(?![A-Z])", normalized.upper())
    if isolated:
        return isolated[-1]


    llm_client = llm if llm is not None else _build_default_llm()
    chain = llm_client.with_structured_output(ExtractOptionOutput)

    res = chain.invoke([
        ("system", "You extract the final single-choice option letter from medical MCQ answer text."),
        ("human", f"""Extract exactly one final option letter from the text.

Rules:
1) Return only one uppercase letter from A-F.
2) If the text contains explanation, use the final explicit conclusion.
3) If multiple letters appear, pick the one most clearly indicated as final answer.

Text:
{answer_text}
""")
    ])

    option = res.option.strip().upper()
    if option not in {"A", "B", "C", "D", "E", "F"}:
        logging.warning("LLM returned unexpected option: %s", option)
        raise ValueError(f"Invalid extracted option: {option}")

    return option
