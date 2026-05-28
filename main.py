from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import httpx
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    """Take ATG screenshot, return parsed horses via Claude vision."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY saknas i Railway Variables")

    image_data = await file.read()
    b64 = base64.b64encode(image_data).decode("utf-8")
    media_type = file.content_type or "image/jpeg"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64}
                        },
                        {
                            "type": "text",
                            "text": """Läs av denna startlista från ATG och returnera EXAKT detta JSON, inga backticks, ingen annan text:

{"raceName":"Lopp X eller loppets namn","horses":[{"number":1,"name":"HÄSTNAMN","driver":"Förnamn Efternamn","odds":1.09}]}

Om bilden inte innehåller ett travlopp: {"error":"Inget lopp hittades"}"""
                        }
                    ]
                }]
            }
        )
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Claude API fel: {r.text}")
        
        data = r.json()
        text = next((b["text"] for b in data["content"] if b["type"] == "text"), "")
        clean = text.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            parsed = json.loads(clean)
            return parsed
        except:
            raise HTTPException(status_code=422, detail=f"Kunde inte tolka svar: {text}")

class AnalyzeRequest(BaseModel):
    raceName: str
    horses: str

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY saknas")
    
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": f"""Du är en professionell svensk travanalytiker.

Lopp: {req.raceName}
{req.horses}

Analysera på svenska, max 200 ord:
1. Är favoriten rimligt prissatt eller överspelad?
2. Finns värdehästar (låg poolandel relativt rimlig chans)?
3. Kommentera starka kuskar
4. Konkret rekommendation: Vinnare / Tvilling / Undvik
Var direkt och konkret som en erfaren spelare till nybörjare."""}]
            }
        )
        data = r.json()
        text = next((b["text"] for b in data["content"] if b["type"] == "text"), "Kunde inte analysera.")
        return {"analysis": text}

# Serve frontend
if os.path.exists("index.html"):
    @app.get("/")
    async def root():
        return FileResponse("index.html")
