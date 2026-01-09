import logging
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    QuestionAnsweringModelOutput,
)
from transformers.utils import ModelOutput

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.core.tp.loss import maybe_vocab_parallel_cross_entropy
from nanorlhf.nanotron.distributed.mpu import MPU

logger = logging.getLogger(__name__)


def is_causal_lm(model):
    """
    Check if the model is a causal language model.

    Args:
        model (nn.Module): The Hugging Face model to check.

    Returns:
        bool: True if the model is a causal language model, False otherwise.
    """
    class_name = model.__class__.__qualname__
    return class_name.endswith("CausalLM") or class_name.endswith("LMHeadModel")


def run_layer(
    layer: nn.Module,
    inputs: Dict[str, Any],
    input_param_name: str,
    gradient_checkpointing_enable: bool,
) -> torch.Tensor:
    """
    Run a model layer with optional gradient checkpointing.

    Args:
        layer (nn.Module): The model layer to run.
        inputs (Dict[str, Any]): The input arguments for the layer.
        input_param_name (str): The name of the primary input parameter.
        gradient_checkpointing_enable (bool): Whether to enable gradient checkpointing.

    Returns:
        torch.Tensor: The output of the layer.
    """
    hidden_states = inputs[input_param_name]
    if gradient_checkpointing_enable:

        def fn(hs, **kwargs):
            return layer(**{input_param_name: hs, **kwargs})

        kwargs = {k: v for k, v in inputs.items() if k != input_param_name}
        output = checkpoint(fn, hidden_states, **kwargs, use_reentrant=False)
    else:
        output = layer(**inputs)

    if isinstance(output, (list, tuple)):
        output = output[0]
    return output


def get_output_type(model: nn.Module):
    """
    Get the appropriate output type for the given Hugging Face model.

    Args:
        model (nn.Module): The Hugging Face model.

    Returns:
        Type[ModelOutput]: The corresponding ModelOutput subclass.
    """
    class_name = model.__class__.__qualname__
    if is_causal_lm(model):
        return CausalLMOutputWithPast
    elif class_name.endswith("SequenceClassification"):
        return SequenceClassifierOutputWithPast
    elif class_name.endswith("TokenClassification"):
        return TokenClassifierOutput
    elif class_name.endswith("QuestionAnswering"):
        return QuestionAnsweringModelOutput
    else:
        return BaseModelOutputWithPast


def post_process_hf_model(
    model: nn.Module,
    mpu: MPU,
    logits: torch.Tensor,
    payload: Dict[str, Any],
    tp_mode: ParallelMode = ParallelMode.TENSOR,
) -> ModelOutput:
    """
    Post-process the outputs of a Hugging Face model based on the presence of labels
    and the model type.

    Args:
        model (nn.Module): The Hugging Face model.
        mpu (MPU): The model parallel unit for distributed operations.
        logits (torch.Tensor): The logits output from the model.
        payload (Dict[str, Any]): The input payload containing user inputs and other info.
        tp_mode (ParallelMode): The tensor parallel mode for distributed operations.

    Returns:
        ModelOutput: The processed model output.
    """
    input_ids = payload["user_inputs"].get("input_ids", None)
    labels = payload["user_inputs"].get("labels", None)
    last_hidden_state = payload.get("hidden_states", None)
    past_key_values = payload["module_list_kwargs"].get("past_key_values", None)

    if logits is None:
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_state, past_key_values=past_key_values)

    config = model.config
    class_name = model.__class__.__qualname__
    batch_size = logits.shape[0]

    if labels is None:
        output_type = get_output_type(model)
        if output_type == TokenClassifierOutput:
            return output_type(logits=logits, hidden_states=last_hidden_state)
        elif output_type == QuestionAnsweringModelOutput:
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            return output_type(start_logits=start_logits, end_logits=end_logits, hidden_states=last_hidden_state)
        else:
            return output_type(logits=logits, hidden_states=last_hidden_state, past_key_values=past_key_values)

    if is_causal_lm(model):
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)
        shift_logits = logits.view(-1, logits.size(-1))
        loss = maybe_vocab_parallel_cross_entropy(shift_logits, shift_labels, mpu, tp_mode)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, hidden_states=last_hidden_state, past_key_values=past_key_values  # noqa
        )
    elif class_name.endswith("SequenceClassification"):
        if config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning(
                f"{class_name} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        num_labels = config.num_labels
        if config.problem_type is None:
            if num_labels == 1:
                config.problem_type = "regression"
            elif num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                config.problem_type = "single_label_classification"
            else:
                config.problem_type = "multi_label_classification"
        labels = labels.to(pooled_logits.device)
        if config.problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif config.problem_type == "single_label_classification":
            loss = nn.functional.cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1))
        elif config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
        else:
            raise RuntimeError(f"Invalid problem type: {config.problem_type}")

        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled_logits, hidden_states=last_hidden_state, past_key_values=past_key_values  # noqa
        )
    elif class_name.endswith("TokenClassification"):
        labels = labels.view(-1).to(logits.device)
        shift_logits = logits.view(-1, config.num_labels).float()
        loss = torch.nn.functional.cross_entropy(shift_logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=last_hidden_state)  # noqa
    elif class_name.endswith("QuestionAnswering"):
        total_loss = None
        start_positions = payload["user_inputs"].get("start_positions", None)
        end_positions = payload["user_inputs"].get("end_positions", None)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            start_loss = torch.nn.functional.cross_entropy(start_logits, start_positions)
            end_loss = torch.nn.functional.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=last_hidden_state
        )
    else:
        raise NotImplementedError(
            f"Using model class `{class_name}` with `labels` is not supported yet. "
            f"Currently supported classes which can be used with `labels` are: "
            "`CausalLM`, `LMHeadModel`, `SequenceClassification`, `TokenClassification`, "
            "and `QuestionAnswering`."
        )
