import logging
from collections import defaultdict
from typing import TypedDict, Dict, List
from output import (
    AgentOutput, AggregatorOutput, AgentEvalOutput,
    AgentErrorDiagnosisOutput, RolePromptOptOutput,
    AggEvalOutput, AggErrorDiagnosisOutput, AggPromptOptOutput
)


class AgentEvalBuffer:
    """Store train data and compressed failure statistics for one node."""
    def __init__(self, max_examples_per_category=5):
        self.total_sample_nums = 0
        self.score = []
        self.max_examples = max_examples_per_category
        self.category_stats = defaultdict(lambda: {
            "count": 0,
            "examples": []
        })
        self.debugflag = True

    def update(self, eval_result):
        self.total_sample_nums += 1
        self.score.append(eval_result.score)

        if eval_result.correctness == "correct":
            return
        category = eval_result.failure_category
        if category == "NONE":
            return

        bucket = self.category_stats[category]
        bucket["count"] += 1

        if len(bucket["examples"]) < self.max_examples:
            bucket["examples"].append(eval_result.explanation)

    def summary_for_llm(self):
        """What will be sent to LLM (bounded size) in XML format."""
        xml_output = "<failure_analysis>\n"

        for category, info in self.category_stats.items():
            xml_output += f'  <category name="{category}">\n'
            xml_output += f'    <count>{info.get("count", 0)}</count>\n'
            xml_output += f'    <examples>\n'

            for ex in info.get("examples", []):
                xml_output += f'      <example>\n{ex}\n      </example>\n'

            xml_output += f'    </examples>\n'
            xml_output += f'  </category>\n'

        xml_output += "</failure_analysis>"
        if self.debugflag:
            self.debugflag = False
            logging.info(f"input to agent diagnosis:\n{xml_output}")

        return xml_output


class AgentEvalManager:
    def __init__(self):



class AggEvalManager:
    """管理 agg_credits"""
    def __init__(self, num_rounds=3):
        # todo: not init as 100
        self.agg_credits = [5.0] * num_rounds
        self.agg_error_buffers = [[] for _ in range(num_rounds)]
        self.need_opt = [False for _ in range(num_rounds)]
        self.learn_rate = 0.2
        self.optimize_threshold = 3.5
        self.min_errors_before_opt = 15
        self.debugflag = True

    def update_and_check(self, agg_idx: int, eval_result):
        # 计算信用分
        old_credit = self.agg_credits[agg_idx]
        new_credit = (1.0 - self.learn_rate) * old_credit + self.learn_rate * float(eval_result.score)

        self.agg_credits[agg_idx] = new_credit

        # compress error buffer
        if eval_result.correctness == "correct":
            return
        category = eval_result.failure_category
        if category == "NONE":
            return

        # 提取选项
        # correct_answer = state["correct_answer"]
        # correct_opt = extract_option(correct_answer)
        # agent_opts = [extract_option(v.get('answer', '')) for v in state["responses"].values()]

        # todo: 如果所有 Agent 全错，不进行处罚/优化
        # all_wrong = all(opt != correct_opt for opt in agent_opts)
        # if all_wrong:
        #     return False

        if len(self.agg_error_buffers[agg_idx]) < 10:
            self.agg_error_buffers[agg_idx].append(eval_result)

        # 触发判断
        if new_credit < self.optimize_threshold and len(self.agg_error_buffers[agg_idx]) >= self.min_errors_before_opt:
            self.need_opt[agg_idx] = True

    def summary_for_llm(self, agg_idx: int):
        xml_output = "<failure_analysis>\n"

        for result in self.agg_error_buffers[agg_idx]:
            xml_output += f'    <category>{result.get("failure_category", "")}</category>\n'
            xml_output += f'    <description>{result.get("explanation", "")}</description>\n'

        xml_output += "</failure_analysis>"
        if self.debugflag:
            self.debugflag = False
            logging.info(f"input to agg diagnosis:\n{xml_output}")

        return xml_output


class DebateState(TypedDict):
    question: str
    correct_answer: str  # 评估时需要 ground truth
    responses: Dict[str, Dict[str, str]]
    final_answer: str

    # 存储各个 Agent 的多轮评估结果、错误解释等
    agent_evaluations: Dict[str, AgentEvalBuffer]
    agent_error_summaries: Dict[str, str]
    agent_prompts: Dict[str, str]  # 存储当前 agent 的 old_prompt

    # Aggregator 的评估与优化状态
    agg_evaluations: AggEvalManager
    agg_error_diagnosis: str  # agg 的优化是串行的，这里共用同一个状态属性
    agg_prompts: List[str]


def create_debate_node(agent_id: str, llm, system_prompt):
    def node(state: DebateState):
        # 提取“其他”agent 的回答
        others = []
        for k, v in state["responses"].items():
            if k != agent_id and v:
                others.append(
                    f"{k}:\nAnswer: {v.get('answer', '')}\nReason: {v.get('reason', '')}"
                )
        others_text = "\n\n".join(others) if others else "No previous opinions."

        # 构造 XML 格式输入
        formatted_input = f"<question>\n{state['question']}\n</question>\n"
        formatted_input += f"<other_agent_responses>\n{others_text}\n</other_agent_responses>"

        chain = llm.with_structured_output(AgentOutput)

        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Please provide your medical reasoning and answer.\n\n{formatted_input}")
        ])

        # 更新状态中对应的 Agent 回答
        new_responses = dict(state["responses"])
        new_responses[agent_id] = {
            "answer": res.answer,
            "reason": res.reason
        }
        return {"responses": new_responses}

    return node


def create_aggregator_node(llm, system_prompt):
    def node(state: DebateState):
        all_answers = "\n\n".join([
            f"{k}:\nAnswer: {v.get('answer', '')}\nReason: {v.get('reason', '')}"
            for k, v in state["responses"].items()
        ])

        formatted_input = f"<question>\n{state['question']}\n</question>\n"
        formatted_input += f"<all_agent_answers>\n{all_answers}\n</all_agent_answers>"

        chain = llm.with_structured_output(AggregatorOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Synthesize the final answer.\n\n{formatted_input}")
        ])
        return {"final_answer": res.final_answer}

    return node


def create_agent_eval_node(agent_id: str, llm, base_system_prompt: str):
    """评估单个 Agent 最终回答的正确性与失败模式"""
    instruction_appendix = """

    CRITICAL EVALUATION RULES:
    You must classify the failure_category using EXACTLY ONE of these categories:
    - DOMAIN_MISMATCH: Inappropriate domain perspective (e.g., economic reasoning for a medical question).
    - KNOWLEDGE_DEFICIT: Incorrect or insufficient domain knowledge.
    - MISINTERPRET_QUESTION: Misunderstands key conditions, constraints, or intent.
    - INCOMPLETE_REASONING: Lacks necessary reasoning steps or justification.
    - OVERGENERALIZATION: Relies on generic patterns without addressing case-specific details.
    - MISALIGNED_OBJECTIVE: Addresses a different goal than what the question asks.
    - INSUFFICIENT_JUSTIFICATION: Conclusion may be correct but weakly justified.
    - RANDOM_OR_UNGROUNDED: Arbitrary, speculative, or unsupported.
    - NONE: The answer is fully correct with no notable issues.

    If the answer is correct, choose NONE. Do NOT invent new labels.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        agent_answer_dict = state["responses"].get(agent_id, {})
        agent_answer_text = f"Answer: {agent_answer_dict.get('answer', '')}\nReason: {agent_answer_dict.get('reason', '')}"

        formatted_input = (
            f"<question>\n{state['question']}\n</question>\n"
            f"<correct_answer>\n{state['correct_answer']}\n</correct_answer>\n"
            f"<agent_answer>\n{agent_answer_text}\n</agent_answer>"
        )

        chain = llm.with_structured_output(AgentEvalOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Evaluate the agent's answer based on the medical multiple-choice question.\n\n{formatted_input}")
        ])

        # 将评估结果追加到状态中
        evals = state.get("agent_evaluations", {}).copy()
        if agent_id not in evals:
            evals[agent_id] = AgentEvalBuffer()

        evals[agent_id].update(res)

        return {"agent_evaluations": evals}

    return node


def create_agent_error_diagnosis_node(agent_id: str, llm, base_system_prompt: str):
    """归因分析：总结 Agent 典型的错误原因"""
    instruction_appendix = """

    TASK REQUIREMENTS:
    You must provide an error attribution summary (attr_summary) that combines common 
    characteristics of the provided similar errors and covers all aspects of the failures.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        formatted_input = state.get("agent_evaluations", {}).get(agent_id, AgentEvalBuffer()).summary_for_llm()

        chain = llm.with_structured_output(AgentErrorDiagnosisOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Summarize the typical reasons for the agent's failures.\n\n{formatted_input}")
        ])

        summaries = state.get("agent_error_summaries", {}).copy()
        summaries[agent_id] = res.attr_summary
        return {"agent_error_summaries": summaries}

    return node


def create_role_prompt_opt_node(agent_id: str, llm, base_system_prompt: str):
    """基于错误汇总，诊断并重构 Agent 的角色提示词"""
    instruction_appendix = """

    PROMPT OPTIMIZATION RULES:
    1. Remove invalid or misleading role assumptions.
    2. Rebuild the prompt to provide clearer role alignment, reasoning discipline, and grounding constraints.
    3. Address the specific failures observed in the failure_summary.
    4. Keep the new prompt concise.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        old_prompt = state.get("agent_prompts", {}).get(agent_id, "No old prompt provided.")
        failure_summary = state.get("agent_error_summaries", {}).get(agent_id, None)

        if failure_summary is None:
            logging.warning("Can't optimize role prompt. No failure_summary provided.")
            return {"agent_prompts": old_prompt}

        formatted_input = (
            f"<old_prompt>\n{old_prompt}\n</old_prompt>\n"
            f"<failure_summary>\n{failure_summary}\n</failure_summary>"
        )

        chain = llm.with_structured_output(RolePromptOptOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Diagnose and reconstruct the agent's role prompt.\n\n{formatted_input}")
        ])

        prompts = state.get("agent_prompts", {}).copy()
        prompts[agent_id] = res.new_prompt
        return {"agent_prompts": prompts}

    return node


def create_agg_eval_node(agg_idx: int, llm, base_system_prompt: str):
    """评估 Aggregator 汇总结果的质量"""
    instruction_appendix = """

    EVALUATION CATEGORIES & SCORING RULES:
    You MUST assign a score from 1 to 5 based EXACTLY on these rules:

    1. Base score by error_category:
    - 'NONE' -> 5 (Correct, clear, logically sound)
    - 'AMBIGUITY' -> 4 (Hesitant or fails to resolve conflicts, no factual error)
    - 'MAJORITY_BIAS' -> 3 (Followed majority opinion, ignored a reasonable minority argument)
    - 'CRITICAL_OMISSION' -> 2 (Omitted correct answer or crucial supporting argument present in debate)
    - 'HALLUCINATION' -> 1 (Introduces information not in debate or contradicts facts)

    2. Adjustments:
    - If you quote a correct argument ignored by the aggregator: +1 (Max 5)
    - If you identify the aggregator accepted an incorrect argument: -1 (Min 1)
    - If HALLUCINATION is present: Final score MUST be 1.

    EXPLANATION REQUIREMENT:
    You MUST explicitly quote or summarize the specific correct argument that was ignored, 
    and the specific incorrect argument that was accepted.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        # 组装完整的辩论上下文
        agent_answers_text = "\n\n".join([
            f"Agent [{k}]:\nAnswer: {v.get('answer', '')}\nReason: {v.get('reason', '')}"
            for k, v in state["responses"].items()
        ])

        formatted_input = (
            f"<question>\n{state['question']}\n</question>\n"
            f"<correct_answer>\n{state['correct_answer']}\n</correct_answer>\n"
            f"<agent_debates>\n{agent_answers_text}\n</agent_debates>\n"
            f"<aggregate_answer>\n{state['final_answer']}\n</aggregate_answer>"
        )

        chain = llm.with_structured_output(AggEvalOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Identify the Aggregator's mistakes and score its performance.\n\n{formatted_input}")
        ])

        # 记录汇总评估结果
        evals = state.get("agg_evaluations")
        evals.update_and_check(agg_idx, res)

        return {"agg_evaluations": evals}

    return node


def create_agg_error_diagnosis_node(agg_idx: int, llm, base_system_prompt: str):
    """汇总 Aggregator 的多次失败评估，生成稳定诊断"""
    instruction_appendix = """

    DIAGNOSIS REQUIREMENTS:
    Provide a concise diagnosis (5-8 sentences max) summarizing:
    1. The dominant error patterns across cases.
    2. What the aggregator systematically overweights or ignores.
    3. One or two high-level behavioral corrections (NOT prompt text).
    Do NOT mention specific questions.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        # 将历史多次评估结果喂入诊断模型
        critic_results = state.get("agg_evaluations", [])
        formatted_input = critic_results.summary_for_llm(agg_idx)

        if not formatted_input:
            logging.warning("Can't diagnosis agg. No incorrect evaluations to diagnose.")
            return {"agg_error_diagnosis": None}

        chain = llm.with_structured_output(AggErrorDiagnosisOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Aggregate multiple round critiques into a concise, stable diagnosis.\n\n{formatted_input}")
        ])

        return {"agg_error_diagnosis": res.critic_diagnosis}

    return node


def create_agg_prompt_opt_node(llm, base_system_prompt: str):
    """优化 Aggregator 的系统提示词"""
    instruction_appendix = """

    CRITICAL PROMPT REWRITE RULES:
    1. DO NOT APPEND instructions to the end.
    2. REWRITE and MERGE existing instructions.
    3. IF adding a new rule, REMOVE or CONDENSE an old rule to keep the total length constant.
    4. Output the raw text only, without conversational padding.
    """

    system_prompt = base_system_prompt + instruction_appendix

    def node(state: DebateState):
        old_prompt = state.get("agg_prompts", "No old prompt provided.")
        critic_diagnosis = state.get("agg_error_diagnosis", None)

        if critic_diagnosis is None:
            logging.warning("Can't optimize agg prompt. No critic_diagnosis provided.")
            return {"agg_prompt": old_prompt}

        formatted_input = (
            f"<old_prompt>\n{old_prompt}\n</old_prompt>\n"
            f"<critic_diagnosis>\n{critic_diagnosis}\n</critic_diagnosis>"
        )

        chain = llm.with_structured_output(AggPromptOptOutput)
        res = chain.invoke([
            ("system", system_prompt),
            ("human", f"Rewrite the Aggregator system prompt to fix the diagnosed behaviors.\n\n{formatted_input}")
        ])

        return {"agg_prompt": res.new_prompt}

    return node
