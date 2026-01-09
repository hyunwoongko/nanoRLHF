from math_verify.grader import logger

logger.disabled = True
# Disable timeout warning from math-verify.

from math_verify import parse, verify

from nanorlhf.eval.utils import get_unnormalized_answer


def compute_score(prompt, prediction, reference):
    """
    Computes the reward score by comparing the model's prediction with the reference answer.

    Args:
        prompt (str): The input prompt string.
        prediction (str): The model's predicted answer string.
        reference (str): The ground truth reference answer string.
    """
    if "boxed" not in reference:
        gold_answer = parse("\\boxed{" + str(reference) + "}")
    else:
        gold_answer = parse(reference)

    model_answer = parse(get_unnormalized_answer(prediction))
    reward = float(verify(gold_answer, model_answer))
    return reward
