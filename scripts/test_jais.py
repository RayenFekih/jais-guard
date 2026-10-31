"""Historical free-generation demo; use jaisguard predict for classification."""

import logging
import logging.config
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import config  # noqa: E402
from logger import LOGGING_CONFIG  # noqa: E402

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def prepare_inputs(tokenizer, prompt: str, device: torch.device) -> dict:
    if tokenizer.chat_template:
        logger.info("Using tokenizer chat template")

        messages = [
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.info(f"Formatted prompt:\n{formatted_prompt}")

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
    else:
        logger.warning("No chat template found. Using raw prompt.")

        logger.info(f"Raw prompt:\n{prompt}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

    return {key: value.to(device) for key, value in inputs.items()}


def main() -> None:
    logger.info("Starting Jais 3 inference test")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"PyTorch: {torch.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    logger.info("Loading tokenizer...")
    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )

    logger.info(f"Tokenizer loaded in {time.perf_counter() - start:.2f}s")
    logger.info(f"Chat template available: {bool(tokenizer.chat_template)}")

    logger.info("Loading model...")
    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    logger.info(f"Model loaded in {time.perf_counter() - start:.2f}s")

    if device == "cuda":
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    logger.info(f"Original prompt: {config.prompt}")

    inputs = prepare_inputs(
        tokenizer=tokenizer,
        prompt=config.prompt,
        device=model.device,
    )

    input_tokens = inputs["input_ids"].shape[1]

    logger.info(f"Input tokens: {input_tokens}")
    logger.info("Generating response...")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    generated_tokens = outputs[0][input_tokens:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    output_tokens = len(generated_tokens)

    logger.info("Generation completed")
    logger.info(f"Output tokens: {output_tokens}")
    logger.info(f"Inference time: {elapsed:.2f}s")
    logger.info(f"Generation speed: {output_tokens / elapsed if elapsed > 0 else 0:.2f} tokens/s")

    if device == "cuda":
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    logger.info(f"Generated response:\n{response}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Jais inference failed")
        sys.exit(1)
