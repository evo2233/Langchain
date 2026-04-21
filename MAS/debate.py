import os
import logging
import json
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from nlp_utils import extract_option
from node import (
    DebateState, create_debate_node, create_aggregator_node,
    create_agent_eval_node, create_agent_error_diagnosis_node, create_role_prompt_opt_node,
    create_agg_eval_node, create_agg_error_diagnosis_node, create_agg_prompt_opt_node, AgentEvalManager, AggEvalManager
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

VLLM_API_BASE = os.getenv("VLLM_API_BASE", f"http://127.0.0.1:8085/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "vllm")

llm = ChatOpenAI(
    model="/model",
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_BASE,
    max_tokens=4096,
    temperature=0.3,
    top_p=0.9,
    extra_body={"repetition_penalty": 1.2}
)

EVAL_BASE_PROMPT = "You are an expert evaluator. Please complete the task according to the requirements below."
OPT_BASE_PROMPT = "You are an expert prompt optimizer. Please provide prompts that match the rule below."


# ==========================================
# 1. 图构建工厂 (Graphs Factories)
# ==========================================

def build_forward_eval_graph(agent_prompts: dict, agg_prompts: list):
    """构建包含【3轮前向辩论】和【结果评估】的执行图"""
    workflow = StateGraph(DebateState)

    # --- 节点定义 ---
    # Round 1
    workflow.add_node("r1_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r1_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r1_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r1_agg", create_aggregator_node(llm, agg_prompts[0]))
    workflow.add_node("r1_agg_eval", create_agg_eval_node(0, llm, EVAL_BASE_PROMPT))

    # Round 2
    workflow.add_node("r2_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r2_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r2_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r2_agg", create_aggregator_node(llm, agg_prompts[1]))
    workflow.add_node("r2_agg_eval", create_agg_eval_node(1, llm, EVAL_BASE_PROMPT))

    # Round 3
    workflow.add_node("r3_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r3_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r3_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r3_agg", create_aggregator_node(llm, agg_prompts[2]))
    workflow.add_node("r3_agg_eval", create_agg_eval_node(2, llm, EVAL_BASE_PROMPT))

    # Agent 终局评估
    workflow.add_node("agent1_eval", create_agent_eval_node("agent_1", llm, EVAL_BASE_PROMPT))
    workflow.add_node("agent2_eval", create_agent_eval_node("agent_2", llm, EVAL_BASE_PROMPT))
    workflow.add_node("agent3_eval", create_agent_eval_node("agent_3", llm, EVAL_BASE_PROMPT))

    # --- 边连接 ---
    workflow.set_entry_point("r1_d1")
    workflow.add_edge("r1_d1", "r1_d2")
    workflow.add_edge("r1_d2", "r1_d3")
    workflow.add_edge("r1_d3", "r1_agg")
    workflow.add_edge("r1_agg", "r1_agg_eval")

    workflow.add_edge("r1_agg_eval", "r2_d1")
    workflow.add_edge("r2_d1", "r2_d2")
    workflow.add_edge("r2_d2", "r2_d3")
    workflow.add_edge("r2_d3", "r2_agg")
    workflow.add_edge("r2_agg", "r2_agg_eval")

    workflow.add_edge("r2_agg_eval", "r3_d1")
    workflow.add_edge("r3_d1", "r3_d2")
    workflow.add_edge("r3_d2", "r3_d3")
    workflow.add_edge("r3_d3", "r3_agg")
    workflow.add_edge("r3_agg", "r3_agg_eval")

    # 3轮汇总完毕后，对各个 Agent 进行打分和错误归因收集
    workflow.add_edge("r3_agg_eval", "agent1_eval")
    workflow.add_edge("agent1_eval", "agent2_eval")
    workflow.add_edge("agent2_eval", "agent3_eval")
    workflow.add_edge("agent3_eval", END)

    return workflow.compile()


def build_agent_opt_graph(agent_id: str):
    """构建 Agent 结构优化图 (诊断 -> 优化)"""
    workflow = StateGraph(DebateState)
    workflow.add_node("diag", create_agent_error_diagnosis_node(agent_id, llm, EVAL_BASE_PROMPT))
    workflow.add_node("opt", create_role_prompt_opt_node(agent_id, llm, OPT_BASE_PROMPT))

    workflow.set_entry_point("diag")
    workflow.add_edge("diag", "opt")
    workflow.add_edge("opt", END)
    return workflow.compile()


def build_agg_opt_graph(agg_idx: int):
    """构建 Aggregator 时间优化图 (诊断 -> 优化)"""
    workflow = StateGraph(DebateState)
    workflow.add_node("diag", create_agg_error_diagnosis_node(agg_idx, llm, EVAL_BASE_PROMPT))
    workflow.add_node("opt", create_agg_prompt_opt_node(llm, OPT_BASE_PROMPT))

    workflow.set_entry_point("diag")
    workflow.add_edge("diag", "opt")
    workflow.add_edge("opt", END)
    return workflow.compile()


# ==========================================
# 2. 训练主循环
# ==========================================


def load_mcq_json_for_langgraph(
    path: str,
    q_key: str = "query",
    a_key: str = "gt",
    num: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    加载原有 MCQ JSON 数据，并转换成当前 LangGraph 训练流程可直接消费的数据格式。

    输入（兼容旧数据）:
    [
      {"query": "...", "gt": "C"},
      ...
    ]

    输出（当前 train_workflow 期望）:
    [
      {"question": "...", "correct_answer": "C"},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset: List[Dict[str, str]] = []
    for idx, item in enumerate(data):
        if q_key not in item or a_key not in item:
            raise KeyError(
                f"Sample index {idx} missing required key(s): "
                f"q_key='{q_key}' or a_key='{a_key}'."
            )

        raw_answer = str(item[a_key])
        dataset.append(
            {
                "question": str(item[q_key]),
                "correct_answer": extract_option(raw_answer, llm=llm),
            }
        )

        if num is not None and len(dataset) >= int(num):
            break

    return dataset

def train_workflow(trainset, max_epochs=3):
    # 1. 初始提示词配置
    global_prompts = {
        "agent_1": "You are an economist. Consider the following problem from an economic perspective.",
        "agent_2": "You are an engineer. Analyze and solve the following problem from a technical viewpoint.",
        "agent_3": "You are an ethicist. Reflect on the following issue from an ethical standpoint.",
    }
    agg_origin_prompt = \
        "You are a summarizing agent, used to summarize the agents' responses and provide a final answer."
    agg_prompts = [agg_origin_prompt, agg_origin_prompt, agg_origin_prompt]

    for epoch in range(max_epochs):
        logging.info(f"\n========== Epoch {epoch + 1}/{max_epochs} ==========")

        # 每轮 Epoch 开始，用最新 Prompts 重新编译【前向图】，避免静态 Prompt 写死在 Node 闭包中
        forward_app = build_forward_eval_graph(global_prompts, agg_prompts)

        # 初始化当前 Epoch 的评估缓存
        epoch_agent_evals = AgentEvalManager()
        epoch_agg_evals = AggEvalManager()

        for i, ex in enumerate(trainset):
            logging.info(f"--- Training Example {i + 1}/{len(trainset)} ---")

            # 挂载单题的初始状态
            state: DebateState = {
                "question": ex["question"],
                "correct_answer": ex["correct_answer"],
                "responses": {"agent_1": {}, "agent_2": {}, "agent_3": {}},
                "final_answer": "",
                "agent_evaluations": epoch_agent_evals,
                "agent_error_summaries": {},
                "agent_prompts": global_prompts.copy(),
                "agg_evaluations": epoch_agg_evals,
                "agg_error_diagnosis": "",
                "agg_prompts": agg_prompts
            }

            # --- A. 执行前向辩论与评估 ---
            state = forward_app.invoke(state)
            logging.info(f"Question completed. Final Aggregated Answer: {state['final_answer'][:50]}...")

            # 累积当前状态的 Evaluation 供后续分析
            epoch_agent_evals = state["agent_evaluations"]
            epoch_agg_evals = state["agg_evaluations"]

            # --- B. Temporal Optimization (Aggregator 实时更新) ---
            for agg_idx, _ in enumerate(agg_prompts):
                if epoch_agg_evals.need_opt[agg_idx]:
                    logging.info("Aggregator performance is poor. Triggering Temporal Optimization...")
                    agg_opt_app = build_agg_opt_graph(agg_idx)

                    state = agg_opt_app.invoke(state)
                    # 更新全局变量
                    agg_prompts = state["agg_prompts"]
                    epoch_agg_evals.need_opt[agg_idx] = False
                    logging.info(f"[Updated Aggregator Prompt]:\n{agg_prompts[agg_idx]}\n")

                    # 使新的 Aggregator Prompt 立即在下一道题生效
                    forward_app = build_forward_eval_graph(global_prompts, agg_prompts)

        # --- C. Spatial Optimization (Agent 周期更新) ---
        logging.info("\nEnd of Epoch. Triggering Spatial Optimization for Agents...")

        epoch_agent_evals.refresh_need_opt()
        for agent_id in ["agent_1", "agent_2", "agent_3"]:
            if epoch_agent_evals.need_opt.get(agent_id, False):
                logging.info(f"Agent [{agent_id}]. Optimizing...")
                agent_opt_app = build_agent_opt_graph(agent_id)

                opt_state: DebateState = {
                    "question": "", "correct_answer": "", "responses": {}, "final_answer": "",
                    "agent_evaluations": epoch_agent_evals,
                    "agent_error_summaries": {},
                    "agent_prompts": global_prompts.copy(),
                    "agg_evaluations": epoch_agg_evals, "agg_error_diagnosis": "", "agg_prompts": agg_prompts
                }
                opt_state = agent_opt_app.invoke(opt_state)

                # 应用更新
                global_prompts[agent_id] = opt_state["agent_prompts"][agent_id]
                logging.info(f"[Updated Agent {agent_id} Prompt]:\n{global_prompts[agent_id]}\n")

    logging.info("Training Completed.")
    return global_prompts, agg_prompts


if __name__ == '__main__':
    # 简单的 Mock 数据格式，你需要将其替换为你真实的 trainset 格式
    dummy_trainset = [
        {
            "question": "Which of the following defines the pressure in the vascular system in the absence of blood flow?\n(A) Pulse pressure\n(B) Critical closing pressure\n(C) Mean circulatory filling pressure\n(D) Perfusion pressure",
            "correct_answer": "C"
        },
        {
            "question": "What is the primary mechanism of action for penicillin?\n(A) Protein synthesis inhibition\n(B) Cell wall synthesis inhibition\n(C) DNA gyrase inhibition\n(D) RNA polymerase inhibition",
            "correct_answer": "B"
        }
    ]

    # 启动训练
    final_agent_prompts, final_agg_prompt = train_workflow(dummy_trainset, max_epochs=3)

    print("\n\n=== Final Optimized Prompts ===")
    print(f"Aggregator:\n{final_agg_prompt}\n")
    for k, v in final_agent_prompts.items():
        print(f"{k}:\n{v}\n")
