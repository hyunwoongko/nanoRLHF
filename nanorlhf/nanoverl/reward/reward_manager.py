import importlib

from transformers import AutoTokenizer


class RewardManager:
    """
    Manages the loading and computation of reward functions for reinforcement learning.

    Args:
        config: Configuration object containing reward function settings.
    """

    def __init__(self, config):
        self.config = config
        self.reward_fn = self.load_reward_fn()
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.tokenizer_name_or_path)

    def compute_score(self, prompt_tokens_unpacked, response_tokens_unpacked):
        """
        Computes reward scores for a batch of prompt-response pairs.

        Args:
            prompt_tokens_unpacked (list): List of tokenized prompts.
            response_tokens_unpacked (list): List of tokenized responses.
        """
        reward_fn_inputs = [
            {
                "prompt_str": self.tokenizer.decode(prompt["input_ids"][0], skip_special_tokens=False),
                "response_str": self.tokenizer.decode(response["input_ids"][0], skip_special_tokens=False),
                "reward_model": response["reward_model"],
            }
            for prompt, response in zip(prompt_tokens_unpacked, response_tokens_unpacked)
        ]

        return self.reward_fn(reward_fn_inputs)

    def load_reward_fn(self):
        """
        Dynamically loads the reward function specified in the configuration.

        Returns:
            function: The loaded reward function.
        """
        try:
            reward_fn_module = importlib.import_module(self.config.reward.path)
            reward_fn = getattr(reward_fn_module, self.config.reward.name)
            return reward_fn
        except ModuleNotFoundError as e:
            raise ImportError(f"Could not import reward function module at {self.config.reward.path}") from e
        except AttributeError as e:
            raise ImportError(
                f"Reward function {self.config.reward.name} not found in module {self.config.reward.path}"
            ) from e
