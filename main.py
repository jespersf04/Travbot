from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ATG_BASE = "https://www.atg.se/services/racinginfo/v1/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    "Referer": "https://www.atg.se/",
    "Origin": "https://www.atg.se",
}

async def atg_get(path: str):
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(f"{ATG_BASE}{path}", headers=HEADERS)
        r.raise_for_status()
        return r.json()

@app.get("/api/today")
async def get_today():
    try:
        return await atg_get("/games/today")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    try:
        return await atg_get(f"/games/{game_id}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/race/{race_id}")
async def get_race(race_id: str):
    try:
        return await atg_get(f"/races/{race_id}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

class AnalyzeRequest(BaseModel):
    raceName: str
    horses: str

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY saknas")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
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
            r.raise_for_status()
            data = r.json()
            text = next((b["text"] for b in data["content"] if b["type"] == "text"), "Kunde inte analysera.")
            return {"analysis": text}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

# Serve frontend
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
elif os.path.exists("index.html"):
    @app.get("/")
    async def root():
        return FileResponse("index.html")
