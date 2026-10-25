from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BatchEncoding,
    AutoTokenizer,
    AutoModelForCausalLM)
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_qwen() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer: PreTrainedTokenizerBase = (
        AutoTokenizer.from_pretrained(MODEL_NAME))
    model: PreTrainedModel = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME))
    return ((tokenizer, model))


def generate_answer(query: str, sources: list[tuple[str, str]],
                    tokenizer: PreTrainedTokenizerBase,
                    model: PreTrainedModel) -> str:
    system_context: str = (
        "You answer questions about the vLLM"
        " codebase using only the provided context. The answer should"
        " not be longer than one sentence. If examples or commands are "
        "in the answer, write it on the same sentence line. "
        "If the context does not contain the answer, say you don’t know")
    user_context: str = "Context:\n"
    for n, (path, text) in enumerate(sources):
        user_context += f"[{n}] {path}\n{text}\n\n"
    user_context += f"Question: {query}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_context},
    ]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False)
    if not isinstance(ids, BatchEncoding):
        raise TypeError("unexpected tokenizer output")
    output = model.generate(
        **ids, max_new_tokens=128)  # type: ignore[operator]
    if not isinstance(output, torch.Tensor):
        raise TypeError("unexpected generation output")
    answer: str = tokenizer.decode(
        output[0][ids["input_ids"].shape[1]:],
        skip_special_tokens=True)  # type: ignore[assignment]
    return answer
