from typing import Literal
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    answer: str = Field(description="The agent's updated answer")
    reason: str = Field(description=" Detailed reasoning about why you choose this option.")


class AggregatorOutput(BaseModel):
    final_answer: str = Field(description="The final consolidated answer after reconciling all opinions.")


class AgentEvalOutput(BaseModel):
    correctness: Literal["correct", "incorrect"] = Field(description="Whether the answer is correct.")
    failure_category: Literal[
        "DOMAIN_MISMATCH", "KNOWLEDGE_DEFICIT", "MISINTERPRET_QUESTION",
        "INCOMPLETE_REASONING", "OVERGENERALIZATION", "MISALIGNED_OBJECTIVE",
        "INSUFFICIENT_JUSTIFICATION", "RANDOM_OR_UNGROUNDED", "NONE"
    ] = Field(description="The primary failure type. Choose exactly one.")
    explanation: str = Field(description="1-2 sentences explaining the reasoning quality.")
    score: int = Field(description="Integer score from 1 to 5.", ge=1, le=5)


class AgentErrorDiagnosisOutput(BaseModel):
    """基于大量错误 explanation 汇总典型错误原因"""
    attr_summary: str = Field(
        description="An error attribution summary that combines common characteristics of similar errors and covers "
                    "all aspects of failures."
    )


class RolePromptOptOutput(BaseModel):
    """诊断并重构智能体的 Role Prompt"""
    new_prompt: str = Field(
        description="A corrected and reconstructed role prompt with clearer alignment, reasoning discipline, "
                    "and grounding constraints."
    )


class AggEvalOutput(BaseModel):
    correctness: Literal["correct", "incorrect"] = Field(description="Whether the aggregate answer is correct.")
    failure_category: Literal["CRITICAL_OMISSION", "MAJORITY_BIAS", "AMBIGUITY", "HALLUCINATION", "NONE"] = Field(
        description="The best match category for the aggregator's failure."
    )
    explanation: str = Field(
        description="Detailed diagnosis explicitly quoting the ignored correct arguments or accepted incorrect claims."
    )
    score: int = Field(
        ge=1, le=5,
        description="Integer score (1-5) strictly following the base score and adjustment rules."
    )


class AggErrorDiagnosisOutput(BaseModel):
    """总结多轮批判结果为稳定诊断"""
    critic_diagnosis: str = Field(
        description="A concise diagnosis (5-8 sentences max) summarizing dominant error categories, systematic biases, "
                    "and high-level behavioral corrections."
    )


class AggPromptOptOutput(BaseModel):
    """优化 Aggregator 的系统提示词"""
    new_prompt: str = Field(
        description="The completely rewritten prompt. Must REWRITE and MERGE existing instructions to keep total "
                    "length roughly constant. Raw text only."
    )
