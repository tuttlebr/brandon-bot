import re


START_CONTEXT = "<START_CONTEXT>"
END_CONTEXT = "<END_CONTEXT>"


def extract_context_regex(
    text, start_token=START_CONTEXT, end_token=END_CONTEXT
):
    start_escaped = re.escape(start_token)
    end_escaped = re.escape(end_token)
    pattern = f"(.*?){start_escaped}(.*?){end_escaped}"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        before_text = match.group(1)  # Text before START_TOKEN
        return before_text
    return text
