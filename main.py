from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
import base64
import json
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "claude-sonnet-4-6"  # stable, widely available model

def anthropic_headers(api_key):
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY saknas i miljövariabler"}

    try:
        image_data = await file.read()
        b64 = base64.b64encode(image_data).decode("utf-8")
        media_type = file.content_type or "image/jpeg"
        if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            media_type = "image/jpeg"

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=anthropic_headers(api_key),
                json={
                    "model": MODEL,
                    "max_tokens": 1500,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                            {"type": "text", "text": 'Las av startlistan i bilden. Returnera BARA giltig JSON utan backticks eller annan text:\n{"raceName":"Lopp X","horses":[{"number":1,"name":"NAMN","driver":"Fornamn Efternamn","odds":1.09}]}\nOm ingen startlista syns: {"error":"Inget lopp hittades"}'}
                        ]
                    }]
                }
            )

        # Log everything so we can see what's wrong
        print(f"[SCAN] Anthropic status: {r.status_code}")
        print(f"[SCAN] Anthropic body: {r.text[:500]}")

        if r.status_code != 200:
            return {"error": f"Anthropic svarade {r.status_code}: {r.text[:300]}"}

        data = r.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block["text"]
                break

        clean = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except Exception:
            return {"error": f"Kunde inte tolka AI-svar: {text[:300]}"}

    except Exception as e:
        print(f"[SCAN] EXCEPTION: {traceback.format_exc()}")
        return {"error": f"Serverfel: {str(e)}"}

class AnalyzeRequest(BaseModel):
    raceName: str
    horses: str

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"analysis": "ANTHROPIC_API_KEY saknas i miljövariabler."}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=anthropic_headers(api_key),
                json={
                    "model": MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": f"""Du ar en professionell svensk travanalytiker.

Lopp: {req.raceName}
{req.horses}

Analysera pa svenska, max 200 ord:
1. Ar favoriten rimligt prissatt eller overspelad?
2. Finns vardehastar (lag poolandel relativt rimlig chans)?
3. Kommentera starka kuskar
4. Konkret rekommendation: Vinnare / Tvilling / Undvik"""}]
                }
            )

        print(f"[ANALYZE] Anthropic status: {r.status_code}")
        print(f"[ANALYZE] Anthropic body: {r.text[:500]}")

        if r.status_code != 200:
            return {"analysis": f"Anthropic svarade {r.status_code}: {r.text[:300]}"}

        data = r.json()
        text = "Kunde inte analysera."
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block["text"]
                break
        return {"analysis": text}

    except Exception as e:
        print(f"[ANALYZE] EXCEPTION: {traceback.format_exc()}")
        return {"analysis": f"Serverfel: {str(e)}"}

# Serve frontend
if os.path.exists("index.html"):
    @app.get("/")
    async def root():
        return FileResponse("index.html")
