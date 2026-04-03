"""Prompt formatting for training and evaluation."""


def format_train_prompt(instruction: str, input_text: str, output_text: str) -> str:
    """Format a complete training example with response."""
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text and input_text.strip():
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += f"\n### Response:\n{output_text}"
    return prompt


def format_eval_prompt(instruction: str, input_text: str) -> str:
    """Format an evaluation prompt without the response."""
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text and input_text.strip():
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    return prompt
