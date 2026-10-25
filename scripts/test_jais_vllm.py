import logging
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "IFM/Jais-3-0.9B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    messages = [
        {"role": "user", "content": "ما هي عاصمة تونس؟"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    logger.info("Loading Jais 3 with vLLM...")
    start = time.perf_counter()

    llm = LLM(
        model=MODEL_ID,
        model_impl="transformers",
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        enforce_eager=True,
    )

    logger.info(f"Model loaded in {time.perf_counter() - start:.2f}s")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
    )

    logger.info("Generating response...")
    start = time.perf_counter()

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    elapsed = time.perf_counter() - start
    output = outputs[0].outputs[0]

    logger.info(f"Generated tokens: {len(output.token_ids)}")
    logger.info(f"Inference time: {elapsed:.2f}s")
    logger.info(f"Response:\n{output.text}")


if __name__ == "__main__":
    main()
