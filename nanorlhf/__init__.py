from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nanorlhf.kernels.utils.huggingface import flash_attention_forward


ALL_ATTENTION_FUNCTIONS["nanorlhf_flash_attention"] = flash_attention_forward
