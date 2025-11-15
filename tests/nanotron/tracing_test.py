from typing import List, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM

from nanorlhf.nanotron.utils.tracing import (
    ModelParallelTracer,
    ModelParallelPlan,
    ModuleParallelPlan,
    ModuleType,
    AttentionType,
    SlicingType,
)


def _safe_enum_str(value) -> str:
    if isinstance(value, AttentionType):
        return value.name.lower()
    if isinstance(value, SlicingType):
        return value.name.lower()
    if isinstance(value, ModuleType):
        return value.value
    if value is None:
        return "None"
    return str(value)


def _format_row(name: str, plan: ModuleParallelPlan) -> str:
    module_type_str = _safe_enum_str(plan.module_type)
    fused_str = _safe_enum_str(plan.attention_type)
    slicing_str = _safe_enum_str(plan.slicing_type)
    reversed_flag = str(plan.is_reversed)
    return (
        f"{name:<55} "
        f"type={module_type_str:<12} "
        f"fused={fused_str:<14} "
        f"dir={slicing_str:<11} "
        f"reversed={reversed_flag}"
    )


def _collect_first_block_ids(plan: ModelParallelPlan) -> set:
    if plan.main_module_list is None or len(plan.main_module_list) == 0:
        return set()
    first_block = plan.main_module_list[0]
    ids = set()
    for _, sub in first_block.named_modules():
        ids.add(id(sub))
    return ids


def _gather_plans_for_first_block(
    tracer: ModelParallelTracer, plan: ModelParallelPlan
) -> List[Tuple[str, ModuleParallelPlan]]:
    id_to_name = tracer.id2name
    selected: List[Tuple[str, ModuleParallelPlan]] = []

    first_block_ids = _collect_first_block_ids(plan)

    if plan.embedding_plan is not None:
        name = id_to_name.get(id(plan.embedding_plan.module), "(unknown)")
        selected.append((name, plan.embedding_plan))

    if plan.head_plan is not None:
        name = id_to_name.get(id(plan.head_plan.module), "(unknown)")
        selected.append((name, plan.head_plan))

    for mp in plan.main_module_list_plans:
        mid = id(mp.module)
        if mid in first_block_ids:
            name = id_to_name.get(mid, "(unknown)")
            selected.append((name, mp))

    for mp in plan.pre_module_list_plans:
        name = id_to_name.get(id(mp.module), "(unknown)")
        selected.append((name, mp))

    for mp in plan.post_module_list_plans:
        name = id_to_name.get(id(mp.module), "(unknown)")
        selected.append((name, mp))

    selected.sort(key=lambda t: t[0])
    return selected


def print_first_block_table(model, tracer: ModelParallelTracer):
    traced: ModelParallelPlan = tracer.trace()
    rows = _gather_plans_for_first_block(tracer, traced)
    print("\nTracing only first layer block")
    for name, mp in rows:
        try:
            print(_format_row(name, mp))
        except Exception as e:
            print(f"{name:<55} type=?           fused=?            dir=?         reversed=?   <-- print failed: {e}")


if __name__ == "__main__":
    MODELS = [
        "gpt2",
        "Qwen/Qwen3-0.6B",
        "zary0/gemma-3-270m-it-jp-sft",
        "llamafactory/tiny-random-Llama-3",
        "PJMixers-Archive/tiny-mistral-safetensors",
        "facebook/opt-350m",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/pythia-410m",
        "HuggingFaceTB/SmolLM-135M",
    ]

    torch.set_grad_enabled(False)

    for model_id in MODELS:
        print(f"\n🚀 MODEL: {model_id}\n" + "=" * 120)

        print("\n[AutoModel]\n")
        try:
            if model_id == "gpt2":
                kwargs = {"add_cross_attention": True}
            else:
                kwargs = {}
            base = AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
            tracer = ModelParallelTracer(base)
            print_first_block_table(base, tracer)
        except Exception as e:
            print(f"[AutoModel] Failed to load: {e}")

        print("\n[AutoModelForCausalLM]\n")
        try:
            lm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            tracer = ModelParallelTracer(lm)
            print_first_block_table(lm, tracer)
        except Exception as e:
            print(f"[AutoModelForCausalLM] Failed to load: {e}")
