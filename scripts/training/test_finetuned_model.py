"""
test_finetuned_model.py
-------------------------
Loads the base model + your fine-tuned LoRA adapter together, and runs it
on new excerpts to see if it correctly extracts IV/DV -- including on text
it did NOT see during training, which is the real test of whether it
learned the task or just memorized the 647 training examples.

USAGE (in Colab, after training has finished):
    python3 test_finetuned_model.py --adapter ./braingpt-mistral-lora --model mistral
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}

# A few test excerpts -- mix of styles, NONE of these are from your
# training set, so this checks generalization, not memorization.
TEST_EXCERPTS = [
    "We tested whether maternal depression during pregnancy predicted "
    "attention problems in offspring at age 10, using data from the ABCD cohort.",

    "Higher levels of household chaos were associated with reduced sleep "
    "duration among adolescents in our sample.",

    "This study examined whether neighborhood walkability was related to "
    "children's physical activity levels and body mass index.",

    "We report descriptive statistics on sample demographics; no hypothesis "
    "testing of variable relationships was conducted in this analysis.",
]


def build_prompt(excerpt: str) -> str:
    return (
        "Extract the independent variable(s) (IV) and dependent variable(s) (DV) "
        f"from this excerpt of a research paper.\n\nExcerpt: \"{excerpt}\""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    args = parser.parse_args()

    hf_id = MODEL_REGISTRY[args.model]
    gpu_available = torch.cuda.is_available()

    print(f"Loading base model: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if gpu_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    print(f"Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    print("\n" + "=" * 70)
    print("TESTING ON NEW (UNSEEN) EXCERPTS")
    print("=" * 70)

    for i, excerpt in enumerate(TEST_EXCERPTS, 1):
        prompt = build_prompt(excerpt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,  # deterministic output, easier to judge quality
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Strip the prompt back off so we only see what the model generated
        generated = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

        print(f"\n--- Test {i} ---")
        print(f"Excerpt: {excerpt}")
        print(f"Model output: {generated.strip()}")

    print("\n" + "=" * 70)
    print("Done. Compare each 'Model output' against what you'd expect a")
    print("human annotator to extract. Reasonable IV/DV extraction on these")
    print("NEW excerpts (not in training data) is the real sign it learned")
    print("the task rather than just memorizing the 647 training examples.")


if __name__ == "__main__":
    main()