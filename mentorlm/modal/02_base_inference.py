import modal

image = modal.Image.debian_slim().pip_install("torch", "transformers>=4.51.0", "accelerate")

volume = modal.Volume.from_name("qwen-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen3-4B"
CACHE_DIR = "/cache"

app = modal.App("mentorlm-base-inference")

@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=600,
)
def generate(prompt: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    volume.commit()
    return response


@app.local_entrypoint()
def main():
    prompt = "¿Qué es self-attention? Explícalo de forma sencilla."
    respuesta = generate.remote(prompt)
    print("=== PROMPT ===")
    print(prompt)
    print("=== RESPUESTA ===")
    print(respuesta)
