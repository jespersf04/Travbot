from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
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
async def scan_image(files: List[UploadFile] = File(...)):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY saknas i miljövariabler"}

    try:
        content = []
        for f in files:
            image_data = await f.read()
            b64 = base64.b64encode(image_data).decode("utf-8")
            media_type = f.content_type or "image/jpeg"
            if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                media_type = "image/jpeg"
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})

        content.append({"type": "text", "text": '''Du ser en eller flera bilder fran ATG som tillsammans visar EN startlista. Slat ihop alla hastar fran alla bilder.

Las av SA MYCKET som syns i bilderna. Returnera BARA giltig JSON utan backticks eller annan text, i detta format:
{
  "raceName": "Lopp X",
  "track": "bana om synlig, annars null",
  "distance": "distans i meter om synlig, annars null",
  "startMethod": "Auto eller Volt om synligt, annars null",
  "horses": [
    {
      "number": 1,
      "name": "HASTNAMN",
      "driver": "Fornamn Efternamn",
      "trainer": "tranare om synlig, annars null",
      "odds": 1.09,
      "postPosition": "sparnummer om synligt, annars null",
      "equipment": "utrustningsandring om synlig (t.ex. barfota, amerikansk vagn), annars null"
    }
  ]
}
Om ingen startlista syns: {"error":"Inget lopp hittades"}'''})

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=anthropic_headers(api_key),
                json={
                    "model": MODEL,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": content}]
                }
            )

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
    track: str = ""
    distance: str = ""
    startMethod: str = ""

ANALYST_FRAMEWORK = """Du ar en av Sveriges vassaste kvantitativa travanalytiker. Du analyserar enligt ett professionellt ramverk byggt pa forskning. Anvand webbsokning for att sla upp AKTUELL form, kusk- och tranarstatistik, och eventuella banforhallanden nar det behovs.

MODELLVIKTER (sa har viktar du faktorerna i din bedomning):
- Spar & Faltkontext 31%: startspar, faltets storlek, startmetod (auto/volt), distans, motstandets klassniva. Pa korta distanser ar innerspar en stor fordel.
- Bana & Distans 23%: banspecifik segerfrekvens, distanshistorik, kurvradie, banunderlagets karaktar.
- Kusk & Tranare 18%: segerprocent, kusk-tranar-synergier, kuskens dagsform pa just denna bana.
- Utrustning & Avel 15%: barfota runt om hojer maxhastighet markant men sliter pa hovar; amerikansk vagn ger matbar hastighetsokning sarskilt pa kort distans; huvudlagsforandringar paverkar skarpa och acceleration.
- Pooldynamik 13%: forhallandet mellan din modellerade vinstsannolikhet och den spelade procenten. Har hittar du vardespelen.

VARDEJAKT (Benter-principen): Du jagar hastar dar din kalibrerade vinstsannolikhet ar HOGRE an deras andel av spelpoolen. Da har marknaden underskattat hasten. Favoriter som ar overspelade (stor poolandel men osaker form) spelar du systematiskt emot.

UTRUSTNINGSEFFEKTER att vaga in om de syns:
- Barfota runt om: lattare, snabbare steg, hojer maxfart. Upprepade barfota-starter = slitna hovar (varning).
- Amerikansk vagn (jankarvagn): minskar dragmotstand, hastighetsokning pa kort distans och milebanor.
- Helstangt huvudlag: maximal skarpa i start. Norskt huvudlag: sparar mental energi, adrenalinkick pa upploppet.

Var konkret och skarp. Anvand siffror. Skriv pa svenska."""

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"analysis": "ANTHROPIC_API_KEY saknas i miljövariabler."}

    context = f"Lopp: {req.raceName}"
    if req.track:
        context += f"\nBana: {req.track}"
    if req.distance:
        context += f"\nDistans: {req.distance} m"
    if req.startMethod:
        context += f"\nStartmetod: {req.startMethod}"

    user_prompt = f"""{context}

Startfalt:
{req.horses}

Gor en djupanalys enligt ditt ramverk (max 350 ord). Anvand garna webbsokning for att slta upp aktuell form, kuskstatistik eller banforhallanden om det stARker analysen. Struktur:

1. FAVORITEN: ar oddset rimligt, overspelat eller finns det varde? Vag in kusk, spar och form.
2. VARDESPEL: vilka hastar har modellen hogre tro pa an marknaden? Motivera med spar/kusk/utrustning, inte bara siffror.
3. SPAR & STARTLAGE: kommentera hur starten paverkar pa denna distans/bana.
4. KUSKAR: lyft de starkaste namnen och vad de tillfor.
5. REKOMMENDATION: konkret. Vinnare, Tvilling-forslag, samt vad du undviker och varfor."""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=anthropic_headers(api_key),
                json={
                    "model": MODEL,
                    "max_tokens": 2000,
                    "system": ANALYST_FRAMEWORK,
                    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            )

        print(f"[ANALYZE] Anthropic status: {r.status_code}")
        print(f"[ANALYZE] Anthropic body: {r.text[:500]}")

        if r.status_code != 200:
            return {"analysis": f"Anthropic svarade {r.status_code}: {r.text[:300]}"}

        data = r.json()
        # Collect all text blocks (web search produces multiple)
        parts = []
        for block in data.get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                parts.append(block["text"])
        text = "\n".join(parts) if parts else "Kunde inte analysera."
        return {"analysis": text}

    except Exception as e:
        print(f"[ANALYZE] EXCEPTION: {traceback.format_exc()}")
        return {"analysis": f"Serverfel: {str(e)}"}

# Serve frontend
if os.path.exists("index.html"):
    @app.get("/")
    async def root():
        return FileResponse("index.html")
