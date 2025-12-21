from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extracts the last occurrence of a boxed expression from the input string.

    Args:
        string (str): The input string to search for boxed expressions.

    Returns:
        Optional[str]: The last boxed expression found, or None if none exists.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def get_unnormalized_answer(text: str) -> str:
    """
    Extracts the last boxed expression from the input text.

    Args:
        text (str): The input text to search for boxed expressions.

    Returns:
        str: The last boxed expression found, or "[invalidanswer]" if none exists.
    """
    answer = last_boxed_only_string(text)
    if answer:
        return answer
    else:
        return "[invalidanswer]"
