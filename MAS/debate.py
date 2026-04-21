import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from nlp_utils import extract_option
from node import (
    DebateState, create_debate_node, create_aggregator_node,
    create_agent_eval_node, create_agent_error_diagnosis_node, create_role_prompt_opt_node,
    create_agg_eval_node, create_agg_error_diagnosis_node, create_agg_prompt_opt_node, AgentEvalManager, AggEvalManager,
    load_json_for_langgraph
)

PROMPT_SAVE_PATH = Path(__file__).resolve().parent / "optimized_prompts.json"


def configure_logging(log_name: str = "training_log.txt") -> Path:
    log_path = Path(__file__).resolve().parent / log_name
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.info("Logging initialized. Log file: %s", log_path)
    return log_path


VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8085/v1")
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


def build_test_graph(agent_prompts: Dict[str, str], agg_prompts: List[str]):
    """构建只包含【3轮辩论 + 汇总】的测试图，不包含评估与优化节点。"""
    workflow = StateGraph(DebateState)

    workflow.add_node("r1_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r1_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r1_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r1_agg", create_aggregator_node(llm, agg_prompts[0]))

    workflow.add_node("r2_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r2_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r2_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r2_agg", create_aggregator_node(llm, agg_prompts[1]))

    workflow.add_node("r3_d1", create_debate_node("agent_1", llm, agent_prompts["agent_1"]))
    workflow.add_node("r3_d2", create_debate_node("agent_2", llm, agent_prompts["agent_2"]))
    workflow.add_node("r3_d3", create_debate_node("agent_3", llm, agent_prompts["agent_3"]))
    workflow.add_node("r3_agg", create_aggregator_node(llm, agg_prompts[2]))

    workflow.set_entry_point("r1_d1")
    workflow.add_edge("r1_d1", "r1_d2")
    workflow.add_edge("r1_d2", "r1_d3")
    workflow.add_edge("r1_d3", "r1_agg")

    workflow.add_edge("r1_agg", "r2_d1")
    workflow.add_edge("r2_d1", "r2_d2")
    workflow.add_edge("r2_d2", "r2_d3")
    workflow.add_edge("r2_d3", "r2_agg")

    workflow.add_edge("r2_agg", "r3_d1")
    workflow.add_edge("r3_d1", "r3_d2")
    workflow.add_edge("r3_d2", "r3_d3")
    workflow.add_edge("r3_d3", "r3_agg")
    workflow.add_edge("r3_agg", END)

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
    workflow.add_node("opt", create_agg_prompt_opt_node(agg_idx, llm, OPT_BASE_PROMPT))

    workflow.set_entry_point("diag")
    workflow.add_edge("diag", "opt")
    workflow.add_edge("opt", END)
    return workflow.compile()


# ==========================================
# 2. 训练主循环
# ==========================================

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


def test_workflow(testset, agent_prompts: Dict[str, str], agg_prompts: List[str]) -> float:
    """仅执行辩论与汇总，在测试集上统计准确率。"""
    test_app = build_test_graph(agent_prompts, agg_prompts)

    correct = 0
    total = len(testset)
    for i, ex in enumerate(testset):
        logging.info("--- Testing Example %s/%s ---", i + 1, total)
        state: DebateState = {
            "question": ex["question"],
            "correct_answer": ex["correct_answer"],
            "responses": {"agent_1": {}, "agent_2": {}, "agent_3": {}},
            "final_answer": "",
            "agent_evaluations": AgentEvalManager(),
            "agent_error_summaries": {},
            "agent_prompts": agent_prompts.copy(),
            "agg_evaluations": AggEvalManager(),
            "agg_error_diagnosis": "",
            "agg_prompts": agg_prompts,
        }

        state = test_app.invoke(state)
        pred_option = extract_option(state["final_answer"], llm=llm)
        is_correct = pred_option == ex["correct_answer"]
        correct += int(is_correct)

        logging.info(
            "Test sample done. pred=%s gt=%s correct=%s",
            pred_option,
            ex["correct_answer"],
            is_correct,
        )

    accuracy = correct / total if total else 0.0
    logging.info("[TEST] Accuracy on MedMCQAtest: %.4f (%d/%d)", accuracy, correct, total)
    return accuracy


def save_optimized_prompts(agent_prompts: Dict[str, str], agg_prompts: List[str], path: Path = PROMPT_SAVE_PATH):
    payload = {
        "agent_prompts": agent_prompts,
        "agg_prompts": agg_prompts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Optimized prompts saved to %s", path)


def load_optimized_prompts(path: Path = PROMPT_SAVE_PATH) -> Tuple[Dict[str, str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    agent_prompts = payload.get("agent_prompts", {})
    agg_prompts = payload.get("agg_prompts", [])
    if not isinstance(agent_prompts, dict) or not isinstance(agg_prompts, list) or len(agg_prompts) < 3:
        raise ValueError(f"Invalid prompt file format: {path}")

    return agent_prompts, agg_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Test MAS debate workflow.")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prompt-path", type=str, default=str(PROMPT_SAVE_PATH))
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path)

    if args.mode == "train":
        configure_logging("training_log.txt")
        trainset = load_json_for_langgraph(path="../data/MedMCQ/MedMCQAtrain.json")
        final_agent_prompts, final_agg_prompts = train_workflow(trainset, max_epochs=args.epochs)
        save_optimized_prompts(final_agent_prompts, final_agg_prompts, prompt_path)

        print("\n\n=== Final Optimized Prompts ===")
        for i, p in enumerate(final_agg_prompts):
            print(f"Aggregator {i + 1}:\n{p}")
        for k, v in final_agent_prompts.items():
            print(f"{k}:\n{v}\n")
    else:
        configure_logging("test_log.txt")
        testset = load_json_for_langgraph(path="../data/MedMCQ/MedMCQAtest.json")
        final_agent_prompts, final_agg_prompts = load_optimized_prompts(prompt_path)
        test_workflow(testset, final_agent_prompts, final_agg_prompts)
