# mira_submodel.py

import json
from pathlib import Path
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./exported_model"
MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

BOT_NAME = "MIRA"  # <-- Edit bot name here

_system_prompt = (
    f"You are {BOT_NAME}, an adaptive AI that learns through conversation. "
    "Respond clearly and respectfully."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu")  # CPU only
_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU only
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

def _memory_file(user_id):
    return MEMORY_DIR / f"{user_id}.json"

def load_memory(user_id):
    mf = _memory_file(user_id)
    if mf.exists():
        with open(mf, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(user_id, memory):
    mf = _memory_file(user_id)
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

async def generate_response(user_id: str, user_message: str) -> str:
    memory = load_memory(user_id)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in memory[-12:])

    prompt = f"{_system_prompt}\n{history}\nUser: {user_message}\n{BOT_NAME}:"

    outputs = _generator(prompt, max_new_tokens=150, num_return_sequences=1)
    text = outputs[0]["generated_text"]

    response = text.split(f"{BOT_NAME}:", 1)[-1].strip()

    memory.append({"role": "User", "content": user_message})
    memory.append({"role": BOT_NAME, "content": response})
    memory = memory[-50:]
    save_memory(user_id, memory)

    return response

if __name__ == "__main__":
    import sys

    async def test():
        user = "testuser"
        msg = " ".join(sys.argv[1:]) or "Hello!"
        reply = await generate_response(user, msg)
        print(f"{BOT_NAME}: {reply}")

    asyncio.run(test())
