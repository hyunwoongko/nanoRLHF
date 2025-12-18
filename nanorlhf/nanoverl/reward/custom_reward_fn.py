from nanorlhf.nanoverl.reward.scorer import math_scorer

tasks = {
    "math_rlvr": math_scorer,
}


def compute_score(inputs):
    assert len(inputs) > 0, "Inputs should not be empty"

    scores = []
    for sample in inputs:
        reward_model = sample["reward_model"]
        reward_type = reward_model["reward_type"]
        scorer = tasks.get(reward_type)
        if scorer is None:
            raise ValueError(f"Unsupported reward type: {reward_type}")

        score = scorer.compute_score(
            prediction=sample["response_str"],
            reference=reward_model["ground_truth"],
        )
        scores.append(score)

    return scores
