from nanorlhf.nanoverl.reward.scorer import math_scorer

tasks = {
    "math_rlvr": math_scorer,
}


def compute_score(inputs):
    """
    Computes reward scores for a list of input samples.

    Args:
        inputs (list): A list of dictionaries, each containing:
            - "prompt_str" (str): The prompt string.
            - "response_str" (str): The model's response string.
            - "reward_model" (dict): A dictionary with keys:
                - "reward_type" (str): The type of reward model to use.
                - "ground_truth" (str): The ground truth reference string.

    Returns:
        list: A list of computed scores for each input sample.
    """
    assert len(inputs) > 0, "Inputs should not be empty"

    scores = []
    for sample in inputs:
        reward_model = sample["reward_model"]
        reward_type = reward_model["reward_type"]
        scorer = tasks.get(reward_type)
        if scorer is None:
            raise ValueError(f"Unsupported reward type: {reward_type}")

        score = scorer.compute_score(
            prompt=sample["prompt_str"],
            prediction=sample["response_str"],
            reference=reward_model["ground_truth"],
        )
        scores.append(score)

    return scores
