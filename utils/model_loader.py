"""
model_loader.py
================
Model loading (4-bit quantized, so a 7-8B model fits in a T4's VRAM) plus
a single generate() function that every notebook uses, so you don't
rewrite the same prompt-in/response-out code in every phase.

_LOADED_MODELS caches an already-loaded model in memory for the CURRENT
session only, so re-running a cell doesn't reload it. After a
disconnect/restart of the runtime, it loads again from scratch (normal -
the weights are still cached on Drive, so it's faster than the very
first download).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_LOADED_MODELS = {}


def load_model(model_key: str, model_id: str, cache_dir=None):
    """
    model_key -- short key (e.g. "llama3-8b") used as the in-memory cache key
    model_id  -- full HuggingFace repo id (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
    cache_dir -- Path to a Drive directory for persistent weight caching
    """
    if model_key in _LOADED_MODELS:
        print(f"[model_loader] '{model_key}' is already loaded in this session.")
        return _LOADED_MODELS[model_key]

    print(f"[model_loader] Loading {model_id} (4-bit)... this may take a while.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )

    _LOADED_MODELS[model_key] = (model, tokenizer)
    print(f"[model_loader] '{model_key}' loaded and cached for this session.")
    return model, tokenizer


def unload_model(model_key: str):
    """Free VRAM before loading a different model (important when testing
    multiple 7-8B models within the same session)."""
    if model_key in _LOADED_MODELS:
        model, _ = _LOADED_MODELS.pop(model_key)
        del model
        torch.cuda.empty_cache()
        print(f"[model_loader] '{model_key}' released from VRAM.")


def generate(model, tokenizer, system_prompt, user_prompt, **gen_kwargs):
    """Single prompt-in / response-out function used by every experiment phase."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, **gen_kwargs)
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()
