from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import anthropic
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"status": "ok", "key_found": bool(key), "key_len": len(key)}

@app.post("/translate")
async def translate(data: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Traduis en français: {data.get('text','')}"}]
    )
    return {"translation": msg.content[0].text}
