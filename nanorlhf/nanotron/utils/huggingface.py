import logging
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    QuestionAnsweringModelOutput,
)
from transformers.utils import ModelOutput

from nanorlhf.nanotron.core.tp.loss import maybe_vocab_parallel_cross_entropy
from nanorlhf.nanotron.distributed.mpu import MPU

logger = logging.getLogger(__name__)


def is_causal_lm(model):
    class_name = model.__class__.__qualname__
    return class_name.endswith("CausalLM") or class_name.endswith("LMHeadModel")


def run_layer(layer: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
    """
    Run a single layer with the given inputs and return the hidden states.
    If the layer returns a tuple or list, the first element is assumed to be the hidden states.

    Args:
        layer (nn.Module): The layer to run.
        inputs (Dict[str, Any]): The input arguments for the layer.

    Returns:
        torch.Tensor: The hidden states output by the layer.
    """
    hidden_states = layer(**inputs)
    if isinstance(hidden_states, (list, tuple)):
        hidden_states = hidden_states[0]
        if not torch.is_tensor(hidden_states):
            raise RuntimeError("Layer forward did not return a tensor in first position.")
    return hidden_states


def post_process_hf_model(
    model: nn.Module,
    mpu: MPU,
    logits: torch.Tensor,
    payload: Dict[str, Any],
) -> ModelOutput:
    config = model.config
    class_name = model.__class__.__qualname__
    batch_size = logits.shape[0]

    input_ids = payload["user_inputs"].get("input_ids", None)
    labels = payload["user_inputs"].get("labels", None)
    last_hidden_state = payload.get("hidden_states", None)

    if logits is None:
        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=payload["module_list_kwargs"].get("past_key_values", None),
        )
    elif is_causal_lm(model):
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)
        shift_logits = logits.view(-1, logits.size(-1))
        loss = maybe_vocab_parallel_cross_entropy(shift_logits, shift_labels, mpu)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=last_hidden_state,  # noqa
            past_key_values=payload["module_list_kwargs"].get("past_key_values", None),
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
            loss=loss,
            logits=pooled_logits,
            hidden_states=last_hidden_state,
            past_key_values=payload["module_list_kwargs"].get("past_key_values", None),
        )
    elif class_name.endswith("TokenClassification"):
        labels = labels.view(-1).to(logits.device)
        shift_logits = logits.view(-1, config.num_labels).float()
        loss = torch.nn.functional.cross_entropy(shift_logits, labels)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=last_hidden_state,  # noqa
        )
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
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=last_hidden_state,
        )
    else:
        raise NotImplementedError(
            f"Using model class `{class_name}` with `labels` is not supported yet. "
            f"Currently supported classes which can be used with `labels` are: "
            "`CausalLM`, `LMHeadModel`, `SequenceClassification`, `TokenClassification`, "
            "and `QuestionAnswering`."
        )
