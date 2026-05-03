import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.callbacks.manager import get_openai_callback
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

AGENT_ORIGIN_PROMPTS = {
    "agent_1": "You are an economist. Consider the following problem from an economic perspective.",
    "agent_2": "You are an engineer. Analyze and solve the following problem from a technical viewpoint.",
    "agent_3": "You are an ethicist. Reflect on the following issue from an ethical standpoint.",
}
AGG_ORIGIN_PROMPT = \
    "You are a summarizing agent, used to summarize the agents' responses and provide a final answer."
EVAL_BASE_PROMPT = "You are an expert evaluator. Please complete the task according to the requirements below."
OPT_BASE_PROMPT = "You are an expert prompt optimizer. Please provide prompts that match the rule below."


def resolve_agent_and_round_config(
    base_agent_prompts: Dict[str, str],
    max_agents_per_round: int,
    total_rounds: int,
) -> Tuple[Dict[str, str], List[str]]:
    if max_agents_per_round < 1 or total_rounds < 1:
        raise ValueError("max_agents_per_round and total_rounds must be >= 1")

    # 按需求使用 max(每轮最大参与数, 初始定义的智能体数)
    target_agent_count = max(max_agents_per_round, len(base_agent_prompts))
    resolved_agent_prompts = dict(base_agent_prompts)
    for idx in range(len(base_agent_prompts) + 1, target_agent_count + 1):
        resolved_agent_prompts[f"agent_{idx}"] = (
            "You are a domain expert. Analyze the question carefully and provide concise reasoning."
        )

    agg_prompts = [AGG_ORIGIN_PROMPT] * total_rounds
    return resolved_agent_prompts, agg_prompts


def configure_logging(log_name: str) -> Path:
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    full_log_name = f"{log_name}_{timestamp}.log"
    log_path = log_dir / full_log_name
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

    # Suppress noisy transport-level request logs from model clients.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    root_logger.info("Logging initialized. Log file: %s", log_path)
    return log_path


def load_llm(gpu_id: int):
    VLLM_API_BASE = os.getenv(f"VLLM_API_BASE", f"http://127.0.0.1:808{gpu_id}/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "vllm")

    return ChatOpenAI(
        model="/model",
        api_key=VLLM_API_KEY,
        base_url=VLLM_API_BASE,
        max_tokens=4096,
        temperature=0.3,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.2}
    )


# ==========================================
# 1. 图构建工厂 (Graphs Factories)
# ==========================================

def build_forward_eval_graph(llm, agent_prompts: Dict[str, str], agg_prompts: List[str]):
    """构建包含【多轮前向辩论】和【结果评估】的执行图"""
    workflow = StateGraph(DebateState)
    agent_ids = sorted(agent_prompts.keys(), key=lambda x: int(x.split("_")[-1]))
    rounds = len(agg_prompts)
    for r in range(1, rounds + 1):
        for i, agent_id in enumerate(agent_ids, start=1):
            workflow.add_node(f"r{r}_d{i}", create_debate_node(agent_id, llm, agent_prompts[agent_id]))
        workflow.add_node(f"r{r}_agg", create_aggregator_node(llm, agg_prompts[r - 1]))
        workflow.add_node(f"r{r}_agg_eval", create_agg_eval_node(r - 1, llm, EVAL_BASE_PROMPT))

    eval_nodes = []
    for i, agent_id in enumerate(agent_ids, start=1):
        node_name = f"agent{i}_eval"
        workflow.add_node(node_name, create_agent_eval_node(agent_id, llm, EVAL_BASE_PROMPT))
        eval_nodes.append(node_name)

    workflow.set_entry_point("r1_d1")
    for r in range(1, rounds + 1):
        for i in range(1, len(agent_ids)):
            workflow.add_edge(f"r{r}_d{i}", f"r{r}_d{i+1}")
        workflow.add_edge(f"r{r}_d{len(agent_ids)}", f"r{r}_agg")
        workflow.add_edge(f"r{r}_agg", f"r{r}_agg_eval")
        if r < rounds:
            workflow.add_edge(f"r{r}_agg_eval", f"r{r+1}_d1")

    workflow.add_edge(f"r{rounds}_agg_eval", eval_nodes[0])
    for i in range(len(eval_nodes) - 1):
        workflow.add_edge(eval_nodes[i], eval_nodes[i + 1])
    workflow.add_edge(eval_nodes[-1], END)

    return workflow.compile()


def build_test_graph(llm, agent_prompts: Dict[str, str], agg_prompts: List[str]):
    """构建只包含【多轮辩论 + 汇总】的测试图，不包含评估与优化节点。"""
    workflow = StateGraph(DebateState)
    agent_ids = sorted(agent_prompts.keys(), key=lambda x: int(x.split("_")[-1]))
    rounds = len(agg_prompts)
    for r in range(1, rounds + 1):
        for i, agent_id in enumerate(agent_ids, start=1):
            workflow.add_node(f"r{r}_d{i}", create_debate_node(agent_id, llm, agent_prompts[agent_id]))
        workflow.add_node(f"r{r}_agg", create_aggregator_node(llm, agg_prompts[r - 1]))

    workflow.set_entry_point("r1_d1")
    for r in range(1, rounds + 1):
        for i in range(1, len(agent_ids)):
            workflow.add_edge(f"r{r}_d{i}", f"r{r}_d{i+1}")
        workflow.add_edge(f"r{r}_d{len(agent_ids)}", f"r{r}_agg")
        if r < rounds:
            workflow.add_edge(f"r{r}_agg", f"r{r+1}_d1")
    workflow.add_edge(f"r{rounds}_agg", END)

    return workflow.compile()


def build_agent_opt_graph(llm, agent_id: str):
    """构建 Agent 结构优化图 (诊断 -> 优化)"""
    workflow = StateGraph(DebateState)
    workflow.add_node("diag", create_agent_error_diagnosis_node(agent_id, llm, EVAL_BASE_PROMPT))
    workflow.add_node("opt", create_role_prompt_opt_node(agent_id, llm, OPT_BASE_PROMPT))

    workflow.set_entry_point("diag")
    workflow.add_edge("diag", "opt")
    workflow.add_edge("opt", END)
    return workflow.compile()


def build_agg_opt_graph(llm, agg_idx: int):
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

def save_training_snapshot(
    snapshot_path: Path,
    epoch: int,
    next_example_idx: int,
    global_prompts: Dict[str, str],
    agg_prompts: List[str],
    epoch_agent_evals: AgentEvalManager,
    epoch_agg_evals: AggEvalManager,
):
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "next_example_idx": next_example_idx,
        "global_prompts": global_prompts,
        "agg_prompts": agg_prompts,
        "epoch_agent_evals": epoch_agent_evals.to_dict(),
        "epoch_agg_evals": epoch_agg_evals.to_dict(),
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Training snapshot saved: %s", snapshot_path)


def load_training_snapshot(snapshot_path: Path):
    with open(snapshot_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "epoch": int(payload["epoch"]),
        "next_example_idx": int(payload["next_example_idx"]),
        "global_prompts": payload["global_prompts"],
        "agg_prompts": payload["agg_prompts"],
        "epoch_agent_evals": AgentEvalManager.from_dict(payload.get("epoch_agent_evals", {})),
        "epoch_agg_evals": AggEvalManager.from_dict(payload.get("epoch_agg_evals", {})),
    }


def _build_validation_subset(test_set):
    subset_size = int(len(test_set) * 0.2)
    return test_set[:subset_size]


def _run_prompt_update_validation(
    llm,
    validation_set,
    agent_prompts_for_validation: Dict[str, str],
    agg_prompts_for_validation: List[str],
    epoch_agent_evals: AgentEvalManager,
    epoch_agg_evals: AggEvalManager,
    trigger_source: str,
):
    if not validation_set:
        logging.info('[PROMPT_UPDATE_VALIDATION] skipped because validation set is empty.')
        return

    accuracy = test_workflow(llm, validation_set, agent_prompts_for_validation, agg_prompts_for_validation)

    credit_payload = {
        'trigger_source': trigger_source,
        'validation_accuracy': accuracy,
    }

    if trigger_source.startswith('agent_'):
        credit_payload['agent_risk'] = {
            trigger_source: epoch_agent_evals.agent_risks.get(trigger_source, 0.0)
        }
    elif trigger_source.startswith('aggregator_round_'):
        agg_idx = int(trigger_source.split('_')[-1]) - 1
        if 0 <= agg_idx < len(epoch_agg_evals.agg_credits):
            credit_payload['aggregator_new_credit'] = {
                f'agg_{agg_idx + 1}': epoch_agg_evals.agg_credits[agg_idx]
            }

    logging.info('[PROMPT_UPDATE_VALIDATION] %s', json.dumps(credit_payload, ensure_ascii=False))


def _apply_uniform_error_allocation(
    epoch_agent_evals: AgentEvalManager,
    epoch_agg_evals: AggEvalManager,
) -> None:
    """对照实验：将错误平均分配到所有 Agent 与 Aggregator 轮次。"""
    if epoch_agent_evals.agent_evals:
        agent_ids = list(epoch_agent_evals.agent_evals.keys())
        merged_scores = []
        merged_total = 0
        merged_correct = 0
        merged_stats = {}
        max_examples = None

        for buffer in epoch_agent_evals.agent_evals.values():
            merged_scores.extend(buffer.scores)
            merged_total += buffer.total_sample_nums
            merged_correct += buffer.correct_sample_nums
            max_examples = buffer.max_examples if max_examples is None else max(max_examples, buffer.max_examples)
            for category, info in buffer.category_stats.items():
                bucket = merged_stats.setdefault(category, {"count": 0, "examples": []})
                bucket["count"] += int(info.get("count", 0))
                bucket["examples"].extend(info.get("examples", []))

        if agent_ids:
            avg_total = merged_total // len(agent_ids)
            avg_correct = merged_correct // len(agent_ids)
            for category in merged_stats.values():
                category["count"] = category["count"] // len(agent_ids)
                if max_examples is not None:
                    category["examples"] = category["examples"][:max_examples]

            for agent_id in agent_ids:
                buffer = epoch_agent_evals.agent_evals[agent_id]
                buffer.scores = list(merged_scores)
                buffer.total_sample_nums = avg_total
                buffer.correct_sample_nums = avg_correct
                buffer.category_stats.clear()
                for category, info in merged_stats.items():
                    bucket = buffer.category_stats[category]
                    bucket["count"] = info["count"]
                    bucket["examples"] = list(info["examples"])

    if epoch_agg_evals.agg_credits:
        avg_credit = sum(epoch_agg_evals.agg_credits) / len(epoch_agg_evals.agg_credits)
        merged_errors = []
        need_opt = False
        for idx in range(len(epoch_agg_evals.agg_credits)):
            merged_errors.extend(epoch_agg_evals.agg_error_buffers[idx])
            need_opt = need_opt or bool(epoch_agg_evals.need_opt[idx])

        for idx in range(len(epoch_agg_evals.agg_credits)):
            epoch_agg_evals.agg_credits[idx] = avg_credit
            epoch_agg_evals.agg_error_buffers[idx] = list(merged_errors[:20])
            epoch_agg_evals.need_opt[idx] = need_opt


def train_workflow(
    llm,
    train_set,
    validation_set,
    max_epochs=3,
    snapshot_path: Path = None,
    resume: bool = False,
    optimization_mode: str = "Both",
    control_mode: str = "default",
    max_agents_per_round: int = 3,
    total_rounds: int = 3,
):
    global_prompts, agg_prompts = resolve_agent_and_round_config(
        AGENT_ORIGIN_PROMPTS, max_agents_per_round, total_rounds
    )

    start_epoch = 0
    start_example_idx = 0
    if resume:
        if snapshot_path is None or not snapshot_path.exists():
            raise FileNotFoundError(f"Resume requested but snapshot not found: {snapshot_path}")
        snapshot = load_training_snapshot(snapshot_path)
        start_epoch = snapshot["epoch"]
        start_example_idx = snapshot["next_example_idx"]
        global_prompts = snapshot["global_prompts"]
        agg_prompts = snapshot["agg_prompts"]
        logging.info(
            "Resuming training from epoch=%s, example_idx=%s using snapshot=%s",
            start_epoch + 1, start_example_idx + 1, snapshot_path
        )

    run_temporal_optimization = optimization_mode in {"Both", "Temporal"}
    run_spatial_optimization = optimization_mode in {"Both", "Spatial"}

    with get_openai_callback() as cb:
        for epoch in range(start_epoch, max_epochs):
            logging.info(f"\n========== Epoch {epoch + 1}/{max_epochs} ==========")

            # 每轮 Epoch 开始，用最新 Prompts 重新编译【前向图】，避免静态 Prompt 写死在 Node 闭包中
            forward_app = build_forward_eval_graph(llm, global_prompts, agg_prompts)

            # 初始化当前 Epoch 的评估缓存
            if resume and epoch == start_epoch:
                epoch_agent_evals = snapshot["epoch_agent_evals"]
                epoch_agg_evals = snapshot["epoch_agg_evals"]
                example_range = range(start_example_idx, len(train_set))
            else:
                epoch_agent_evals = AgentEvalManager()
                epoch_agg_evals = AggEvalManager()
                example_range = range(len(train_set))

            for i in example_range:
                ex = train_set[i]
                logging.info(f"--- Training Example {i + 1}/{len(train_set)} ---")

                # 挂载单题的初始状态
                state: DebateState = {
                    "question": ex["question"],
                    "correct_answer": ex["correct_answer"],
                    "responses": {agent_id: {} for agent_id in global_prompts.keys()},
                    "final_answer": "",
                    "agent_evaluations": epoch_agent_evals,
                    "agent_error_summaries": {},
                    "agent_prompts": global_prompts.copy(),
                    "agg_evaluations": epoch_agg_evals,
                    "agg_error_diagnosis": "",
                    "agg_prompts": agg_prompts
                }

                # --- A. 执行前向辩论与评估 ---
                try:
                    state = forward_app.invoke(state)
                except Exception:
                    if snapshot_path is not None:
                        save_training_snapshot(
                            snapshot_path=snapshot_path,
                            epoch=epoch,
                            next_example_idx=i,
                            global_prompts=global_prompts,
                            agg_prompts=agg_prompts,
                            epoch_agent_evals=epoch_agent_evals,
                            epoch_agg_evals=epoch_agg_evals,
                        )
                    logging.exception("Training interrupted at epoch=%s, example=%s", epoch + 1, i + 1)
                    raise
                logging.info(f"Question completed. Final Aggregated Answer: {state['final_answer'][:50]}...")

                # 累积当前状态的 Evaluation 供后续分析
                epoch_agent_evals = state["agent_evaluations"]
                epoch_agg_evals = state["agg_evaluations"]
                if control_mode == "uniform":
                    _apply_uniform_error_allocation(epoch_agent_evals, epoch_agg_evals)

                # --- B. Temporal Optimization (Aggregator 实时更新) ---
                if run_temporal_optimization:
                    for agg_idx, _ in enumerate(agg_prompts):
                        if epoch_agg_evals.need_opt[agg_idx]:
                            logging.info("Aggregator performance is poor. Triggering Temporal Optimization...")
                            agg_opt_app = build_agg_opt_graph(llm, agg_idx)

                            old_agg_prompts = list(agg_prompts)
                            state = agg_opt_app.invoke(state)

                            _run_prompt_update_validation(
                                llm,
                                validation_set,
                                global_prompts,
                                old_agg_prompts,
                                epoch_agent_evals,
                                epoch_agg_evals,
                                trigger_source=f"aggregator_round_{agg_idx + 1}",
                            )

                            # 更新全局变量
                            agg_prompts = state["agg_prompts"]
                            epoch_agg_evals.agg_error_buffers[agg_idx] = []
                            epoch_agg_evals.need_opt[agg_idx] = False
                            epoch_agg_evals.agg_credits[agg_idx] = 3.0
                            logging.info(f"[Updated Aggregator Prompt]:\n{agg_prompts[agg_idx]}\n")

                            # 使新的 Aggregator Prompt 立即在下一道题生效
                            forward_app = build_forward_eval_graph(llm, global_prompts, agg_prompts)
                else:
                    logging.info("Temporal Optimization disabled by optimization_mode=%s", optimization_mode)

                if snapshot_path is not None:
                    next_example_idx = i + 1
                    next_epoch = epoch
                    if next_example_idx >= len(train_set):
                        next_example_idx = 0
                        next_epoch = epoch + 1
                    save_training_snapshot(
                        snapshot_path=snapshot_path,
                        epoch=next_epoch,
                        next_example_idx=next_example_idx,
                        global_prompts=global_prompts,
                        agg_prompts=agg_prompts,
                        epoch_agent_evals=epoch_agent_evals,
                        epoch_agg_evals=epoch_agg_evals,
                    )
            resume = False

            # --- C. Spatial Optimization (Agent 周期更新) ---
            if run_spatial_optimization:
                logging.info("\nEnd of Epoch. Triggering Spatial Optimization for Agents...")

                epoch_agent_evals.refresh_need_opt()
                for agent_id in global_prompts.keys():
                    if epoch_agent_evals.need_opt.get(agent_id, False):
                        logging.info(f"Agent [{agent_id}]. Optimizing...")
                        agent_opt_app = build_agent_opt_graph(llm, agent_id)

                        opt_state: DebateState = {
                            "question": "", "correct_answer": "", "responses": {}, "final_answer": "",
                            "agent_evaluations": epoch_agent_evals,
                            "agent_error_summaries": {},
                            "agent_prompts": global_prompts.copy(),
                            "agg_evaluations": epoch_agg_evals, "agg_error_diagnosis": "", "agg_prompts": agg_prompts
                        }
                        old_agent_prompts = global_prompts.copy()
                        opt_state = agent_opt_app.invoke(opt_state)

                        _run_prompt_update_validation(
                            llm,
                            validation_set,
                            old_agent_prompts,
                            agg_prompts,
                            epoch_agent_evals,
                            epoch_agg_evals,
                            trigger_source=agent_id,
                        )

                        # 应用更新
                        global_prompts[agent_id] = opt_state["agent_prompts"][agent_id]
                        logging.info(f"[Updated Agent {agent_id} Prompt]:\n{global_prompts[agent_id]}\n")
            else:
                logging.info("Spatial Optimization disabled by optimization_mode=%s", optimization_mode)

        logging.info(
            "Training token usage summary: prompt_tokens=%s completion_tokens=%s total_tokens=%s successful_requests=%s total_cost=%s",
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_tokens,
            cb.successful_requests,
            cb.total_cost,
        )

    logging.info("Training Completed.")
    return global_prompts, agg_prompts


def test_workflow(llm, test_set, agent_prompts: Dict[str, str], agg_prompts: List[str]) -> float:
    """仅执行辩论与汇总，在测试集上统计准确率。"""
    test_app = build_test_graph(llm, agent_prompts, agg_prompts)

    correct = 0
    total = len(test_set)
    for i, ex in enumerate(test_set):
        logging.info("--- Testing Example %s/%s ---", i + 1, total)
        state: DebateState = {
            "question": ex["question"],
            "correct_answer": ex["correct_answer"],
            "responses": {agent_id: {} for agent_id in agent_prompts.keys()},
            "final_answer": "",
            "agent_evaluations": AgentEvalManager(),
            "agent_error_summaries": {},
            "agent_prompts": agent_prompts.copy(),
            "agg_evaluations": AggEvalManager(),
            "agg_error_diagnosis": "",
            "agg_prompts": agg_prompts,
        }

        state = test_app.invoke(state)
        pred_option = extract_option(state["final_answer"])
        is_correct = pred_option == ex["correct_answer"]
        correct += int(is_correct)

        logging.info(
            "Test sample done. pred=%s gt=%s correct=%s",
            pred_option,
            ex["correct_answer"],
            is_correct,
        )

    accuracy = correct / total if total else 0.0
    logging.info("[TEST] Accuracy on test set: %.4f (%d/%d)", accuracy, correct, total)
    return accuracy


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path]:
    dataset_dir = Path(__file__).resolve().parent.parent / "data" / dataset
    train_path = dataset_dir / f"{dataset}train.json"
    test_path = dataset_dir / f"{dataset}test.json"
    return train_path, test_path


def save_optimized_prompts(agent_prompts: Dict[str, str], agg_prompts: List[str], path: Path):
    payload = {
        "agent_prompts": agent_prompts,
        "agg_prompts": agg_prompts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("Optimized prompts saved to %s", path)


def load_optimized_prompts(path: Path) -> Tuple[Dict[str, str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    agent_prompts = payload.get("agent_prompts", {})
    agg_prompts = payload.get("agg_prompts", [])
    if not isinstance(agent_prompts, dict) or not isinstance(agg_prompts, list) or len(agg_prompts) < 1:
        raise ValueError(f"Invalid prompt file format: {path}")

    return agent_prompts, agg_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Test MAS debate workflow.")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--no_opt", action="store_true", help="Use no optimize prompt for test mode")
    parser.add_argument("--resume", action="store_true", help="Resume train mode from snapshot.")
    parser.add_argument("--gpu_id", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="MedMCQA")
    parser.add_argument("--max_agents_per_round", type=int, default=3)
    parser.add_argument("--total_rounds", type=int, default=3)
    parser.add_argument(
        "--optimization_mode",
        choices=["Both", "Temporal", "Spatial"],
        default="Both",
        help="Training optimization mode: Both (default), Temporal only, or Spatial only.",
    )
    parser.add_argument(
        "--control_mode",
        choices=["default", "uniform"],
        default="default",
        help="Error attribution mode: default or uniform (control experiment).",
    )
    args = parser.parse_args()

    llm = load_llm(args.gpu_id)
    prompt_path = Path(f"./result/debate_{args.dataset}_{args.optimization_mode}_optimized_prompts.json")
    train_path, test_path = resolve_dataset_paths(args.dataset)

    logging.info("Config: GPU=%s, Opt=%s, Control=%s", args.gpu_id, args.optimization_mode, args.control_mode)
    logging.info(
        "Runtime Args: mode=%s, dataset=%s, epochs=%s, resume=%s, no_opt=%s",
        args.mode, args.dataset, args.epochs, args.resume, args.no_opt,
    )

    if args.mode == "train":
        configure_logging(f"debate_{args.dataset}_train_log")
        train_set = load_json_for_langgraph(path=str(train_path))
        test_set = load_json_for_langgraph(path=str(test_path))
        validation_set = _build_validation_subset(test_set)
        logging.info("Validation set prepared from test set head 20%%: %s/%s", len(validation_set), len(test_set))
        snapshot_path = Path(f"./result/debate_{args.dataset}_{args.optimization_mode}_train_snapshot.json")
        final_agent_prompts, final_agg_prompts = train_workflow(
            llm,
            train_set,
            validation_set=validation_set,
            max_epochs=args.epochs,
            snapshot_path=snapshot_path,
            resume=args.resume,
            optimization_mode=args.optimization_mode,
            control_mode=args.control_mode,
            max_agents_per_round=args.max_agents_per_round,
            total_rounds=args.total_rounds,
        )
        save_optimized_prompts(final_agent_prompts, final_agg_prompts, prompt_path)

        print("\n\n=== Final Optimized Prompts ===")
        for i, p in enumerate(final_agg_prompts):
            print(f"Aggregator {i + 1}:\n{p}")
        for k, v in final_agent_prompts.items():
            print(f"{k}:\n{v}\n")
    else:
        configure_logging(f"debate_{args.dataset}_test_log")
        test_set = load_json_for_langgraph(path=str(test_path))
        if args.no_opt:
            final_agent_prompts, final_agg_prompts = resolve_agent_and_round_config(
                AGENT_ORIGIN_PROMPTS, args.max_agents_per_round, args.total_rounds
            )
        else:
            final_agent_prompts, final_agg_prompts = load_optimized_prompts(prompt_path)
        test_workflow(llm, test_set, final_agent_prompts, final_agg_prompts)
