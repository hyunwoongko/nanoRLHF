from math_verify import parse, verify

from nanorlhf.eval.eval_utils import get_unnormalized_answer


def compute_score(prediction, reference):
    if "boxed" not in reference:
        gold_answer = parse("\\boxed{" + str(reference) + "}")
    else:
        gold_answer = parse(reference)

    model_answer = parse(get_unnormalized_answer(prediction))
    reward = float(verify(gold_answer, model_answer))
    return reward

