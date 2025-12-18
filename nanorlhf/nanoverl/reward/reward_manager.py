import importlib

from transformers import AutoTokenizer


class RewardManager:

    def __init__(self, config):
        self.config = config
        self.reward_fn = self.load_reward_fn()
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor.tokenizer_name_or_path)

    def compute_score(self, response_tokens_unpacked):
        reward_fn_inputs = [
            {
                # shape of input_ids is [1, num_completion_tokens].
                "response_str": self.tokenizer.decode(response["input_ids"][0], skip_special_tokens=True),
                "reward_model": response["reward_model"],
            }
            for response in response_tokens_unpacked
        ]

        return self.reward_fn(reward_fn_inputs)

    def load_reward_fn(self):
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
