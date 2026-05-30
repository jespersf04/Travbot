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

def extract_json(text):
    """Plocka ut forsta giltiga JSON-objektet ur en text, aven om AI:n
    lagt till inledande/avslutande prat eller kodstaket."""
    if not text:
        return None
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except Exception:
            return None
    return None

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

        content.append({"type": "text", "text": '''Du ser EN eller FLERA bilder fran ATG. Bilderna visar OLIKA DELAR av SAMMA startlista for ETT lopp - till exempel hast 1-5 i en bild och hast 7-12 i en annan, eller loppinformationen i en bild och hastarna i en annan.

En bild kan innehalla mycket ovrig text (rubriker som "Vinnare Lopp 3", "Lopp 1 startar 14:30", baninfo, knappar som "Sortera"/"Anpassa", "Sorterad pa startnummer", flikar med loppnummer). IGNORERA allt sadant brus. En bild kan ocksa visa BARA EN ENDA hast med statistik - det ar helt OK, las av den hasten anda. Returnera alltid giltig JSON aven om bara en hast syns.

VIKTIGT: Slat ihop ALLA hastar fran ALLA bilder till EN gemensam lista. Hoppa inte over nagon hast. Om en bild visar loppinfo (bana, distans, startmetod) och en annan visar hastar, kombinera bada. Lista varje hast EN gang, sorterat efter startnummer. Om en hast bara visas delvis (t.ex. bara namn och odds langst ner utan statistikkort), ta anda med den med de falt du ser.

Las av sa mycket som syns. Returnera BARA giltig JSON utan backticks eller annan text, i detta format:
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
      "equipment": "utrustningsandring om synlig (t.ex. barfota, amerikansk vagn), annars null",
      "winPct": "seger% som tal om synligt (t.ex. 60), annars null",
      "placePct": "plats% som tal om synligt (t.ex. 80), annars null",
      "record": "rekordtid om synlig (t.ex. 14,7aM), annars null",
      "formLivs": "livsform om synlig (t.ex. 5: 3-1-0), annars null",
      "form2026": "innevarande ars form om synlig (t.ex. 4: 3-1-0), annars null",
      "form2025": "fjolarsform om synlig, annars null",
      "krPerStart": "kr/start om synligt, annars null",
      "homeTrack": "hemmabana om synlig, annars null",
      "avgOdds": "snittodds om synligt (t.ex. 7,63), annars null"
    }
  ]
}
VIKTIGT om avancerad statistik: Pa ATG kan varje hast expanderas sa ett statistikkort visas med SKOR, VAGN, SPAR, REKORD, PENGAR, LIVS, SEGER%, PLATS%, KR/START, HEMMABANA, SNITTODDS m.m. Om dessa fält syns for en hast, las av dem och fyll i motsvarande falt ovan. Syns de inte for en viss hast, satt null - gissa ALDRIG siffror.
Ignorera hastar markerade som strukna/EJ (overstruket namn). Om ingen startlista syns alls: {"error":"Inget lopp hittades"}'''})

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

        parsed = extract_json(text)
        if parsed is not None:
            return parsed
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

SPELFORMER och NAR de passar (du valjer den form som passar LOPPETS karaktar):
- Vinnare: pricka ettan. Passar nar en hast ar klart bast OCH har hyfsat odds (ej under ~2.5). Lag utdelning.
- Plats: hasten ska komma topp 3 (7+ startande) eller topp 2 (4-6 startande). Passar en stark, palitlig hast vars VINST ar osaker men som nastan sakert ar topp 3. Lagre utdelning, hogre traffchans.
- Tvilling: de tva forsta i valfri ordning. Passar nar TVA hastar sticker ut som klart basta.
- Komb: ettan och tvaan i RATT ordning. Svarare an tvilling, hogre utdelning. Passar bara nar du har en tydlig etta OCH en tydlig tvaa - inte annars.
- Trio: de tre forsta. Kan spelas rakt (exakt ordning) eller som GARDERING (valj 4-5 hastar, alla kombinationer). Trio med gardering passar JAMNA, oppna lopp dar 3-5 hastar ar svara att skilja at - da ger garderingen tackning och hog utdelning. Det ar svaret pa "jamnt lopp".
- Spik: ingen egen form - en hast sa saker att den last i ett kommande V-spel. Foreslas for stenhart favoritlopp dar enskilt spel ger for lite.

SPELDISCIPLIN - MYCKET VIKTIGT:
Du ar en analytiker, inte en tipstjanst. Din uppgift ar att ge ett ARLIGT svar - inte att hitta ett spel. Om inget genuint spelvarde finns: saga det rakt och kortfattat. "Hoppa over" ar ett fullstandigt, professionellt svar.

Utgangslaget ar alltid: INGET SPEL. Du avviker bara fran det nar ALLA dessa kriterier ar uppfyllda:
1. Det finns ett tydligt vardegap (poolandel underskattar verklig chans baserat pa faktisk statistik)
2. Loppets karaktar passar en specifik spelform naturligt
3. Du kan motivera spelet utan att stracja pa resonemang

Om du tvekar - spela inte. Om du maste leta efter ett argument - spela inte. Om faltet ar otydligt och inget sticker ut - spela inte.

- Foreslå max EN spelform per lopp, aldrig fler. 
- Flera lopp utan spelvarde pa rad ar ett BRA utfall - inte ett misslyckande.
- Om INGET lopp under en dag har genuint varde: saga det. "Ingen av dagens lopp erbjuder tillrackligt tydligt varde for att motivera spel" ar ett utmarkt svar.
- Var ARLIG om traffchans vs utdelning: hog utdelning betyder alltid lag sannolikhet. Lova aldrig vinst.

LAS LOPPETS KARAKTAR (gor detta FORST, innan du valjer form):
Ett bra spelval foljer av hur loppet ser ut. Resonera dig fram sa har:
1. Hur stark ar favoriten? Titta pa oddsgapet till tvaan och pa formdata. En hast pa 1.4 med 60% seger% ar en annan sak an en pa 2.8 dar fyra hastar ar tatt bakom.
2. Hur djupt/jamnt ar faltet? Rakna hur manga hastar som realistiskt kan vinna (liknande odds, bra spar, ok form). 2 hastar = grunt. 5+ jamna = djupt/oppet.
3. Var finns vardet? En eller flera hastar dar form/spar motiverar hogre chans an poolandelen visar?

Lat svaret styra formen - och MOTIVERA kopplingen tydligt for anvandaren:
- EN dominant hast (stark form, klart battre an resten), ok odds (>2.5): VINNARE pa den. Ar oddset for lagt (<1.6): SPIK eller hoppa over - inte vart att binda pengar.
- En palitlig hast vars vinst ar osaker men som nastan sakert ar topp 3 (t.ex. hog plats% men tufft motstand om segern): PLATS - tryggare an vinnare, battre odds an man tror.
- EXAKT TVA hastar sticker ut tydligt over resten: TVILLING. Detta ar ett "tvilling-lopp" nar tva ar klart basta men inbordes ordning ar oviss. Forklara att det ar darfor tvilling slar bade vinnare (for osakert vem av de tva) och trio (for tunt under de tva).
- Tydlig etta OCH tydlig tvaa, men du tror dig veta ORDNINGEN: KOMB ger battre betalt an tvilling for samma trafftanke.
- JAMNT, OPPET lopp - 3-5 hastar gar knappt att skilja: TRIO MED GARDERING. Detta ar ett "tajt lopp". Motivera: nar ingen sticker ut ar det dumt att spika eller spela vinnare (for lag traffchans pa ratt ensam hast), men garderad trio tacker utfallen och en skrall ger hog utdelning. Namn de 3-5 hastar du garderar.
- Stenhard favorit + en intressant utmanare till hogt odds: tvilling favorit+utmanare kan vara smart (favoriten ankrar, utmanaren ger utdelningen).

KRITISK REGEL - INGEN TVILLING I OPPNA FALT:
Om du i din karaktarsanalys beskriver faltet som "djupt", "oppet", "minst 3-5 hastar kan vinna", "inte ett tvahastlopp" eller liknande — DA AR TVILLING FEL SVAR. Det vore inkonsekvent. I sa fall MASTE du valja antingen TRIO MED GARDERING (om du ser 3-5 kandidater med varde) eller HOPPA OVER (om inget spel genuint lonar sig). Att landa i tvilling efter att ha beskrivit ett djupt falt ar ett logikfel — fa inte gora det.

Poangen: forklara ALLTID VARFOR loppets form gor just den spelformen smart - inte bara vilken form. Anvandaren ska forsta resonemanget "det blir ett tajt lopp, darfor garderad trio" eller "tva stack ut, darfor tvilling". Om ingen form ger ett genuint bra lage: saga att loppet inte ar spelvart.

UTRUSTNINGSEFFEKTER att vaga in om de syns:
- Barfota runt om: lattare, snabbare steg, hojer maxfart. Upprepade barfota-starter = slitna hovar (varning).
- Amerikansk vagn (jankarvagn): minskar dragmotstand, hastighetsokning pa kort distans och milebanor.
- Helstangt huvudlag: maximal skarpa i start. Norskt huvudlag: sparar mental energi, adrenalinkick pa upploppet.

FAKTA KONTRA BEDOMNING - LAS DETTA NOGA:
Vissa hastar kan ha RIKTIG statistik fran ATG med (seger%, plats%, rekord, form Livs/2026/2025, kr/start, hemmabana, snittodds). Detta ar FAKTISKA siffror, inte gissningar.
- Nar en hast HAR sadan statistik: anvand den som din primara faktagrund. Lat seger%, form och rekord vaga TYNGRE an dina egna antaganden. En hast med hog seger% och stark form har bevisad kapacitet; en med lag seger% trots bra spar/kusk ar en varningssignal.
- Tolka ALLTID dessa fakta GENOM ramverket: vag in spar, kusk, utrustning och pooldynamik enligt modellvikterna. Statistiken sager VAD hasten gjort; ramverket hjalper dig forsta OM det haller i just detta lopp.
- Nar en hast SAKNAR statistik (fält ar null): sag tydligt att din bedomning ar en UPPSKATTNING baserad pa spar och kusk, inte pa formdata.
- Implicit sannolikhet (1/odds) och poolandel ar faktiska berakningar - dem kan du lita pa.
- "Modellvarde" och "edge" du sjalv satter ar DINA bedomningar. Markera dem som bedomningar, inte fakta. Hitta ALDRIG pa seger%, banstatistik eller spar-segerprocent - om du inte har en verifierad siffra, sag att det ar en strukturell bedomning.
- I rekommendationen: ange tydligt for varje hast om radet vilar pa FAKTISK statistik (stark grund) eller pa STRUKTURELL bedomning (svagare grund). Anvandaren ska veta skillnaden.

Var konkret och skarp. Skilj alltid fakta fran bedomning. Skriv pa svenska."""

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

Gor en djupanalys enligt ditt ramverk (max 400 ord). Flera hastar kan ha riktig ATG-statistik med (seger%, plats%, rekord, form, kr/start, snittodds) - anvand den som faktagrund och tolka den genom ramverket. Anvand garna webbsokning for aktuell form eller kuskstatistik om det starker analysen. Struktur:

1. FAVORITEN: ar oddset rimligt, overspelat eller finns varde? Om hasten har seger%/form-data, utga fran den. Vag in kusk, spar och form.
2. VARDESPEL: vilka hastar har hogre verklig chans an marknaden tror? Prioritera hastar dar RIKTIG statistik (bra seger%/form) mojter ett hogre odds an formen motiverar. Motivera med spar/kusk/utrustning OCH faktisk statistik nar den finns.
3. SPAR & STARTLAGE: hur paverkar starten pa denna distans/bana? Var arlig: om du inte har verifierad spar-statistik, sag att det ar en strukturell bedomning.
4. KUSKAR: lyft de starkaste namnen och vad de tillfor.
5. REKOMMENDATION: Borja med att LASA LOPPETS KARAKTAR i 1-2 meningar: hur stark ar favoriten, hur jamnt ar faltet, var finns eventuellt varde? Lat den lasningen STYRA beslutet - och det ar helt OK att beslutet ar "inget spel". Forklara varfor formen passar loppet, eller varfor inget spel ar motiverat. MAX en spelform, aldrig fler. Om du tvekar - hoppa over.

Avsluta med en av dessa slutrader:
"DAGENS SPEL: [spelform] [hastnummer/namn] — [en mening varfor]. Kalibrerad traffchans ~X%, breakeven-odds ~Y (1/X). Lägg spelet om [spelform]odds är över Y när du lägger det — under det är väntevärdet negativt."
ELLER:
"DAGENS SPEL: Hoppa over — [en mening varfor]. Kalibrerad traffchans basta kandidat ~X%, breakeven-odds ~Y. Aktuellt odds Z ar [over/under] breakeven."

KRITISKT: Hoppa-over-raden MASTE alltid innehalla breakeven-odds for den basta kandidaten i loppet. Annars vet anvandaren inte om beslutet ar matematiskt motiverat. Aven nar du hoppar over ska du visa matten.

Hur du raknar och avgör:
- Traffchans for vinnare: din kalibrerade sannolikhet (baserat pa seger%, poolandel, spar, kusk)
- Traffchans for tvilling: (sannolikhet A × sannolikhet B) × 2, grovt
- Breakeven-odds: 1 / traffchans. Over detta = positivt vantevarde, under = negativt.
- REGEL: rekommendera spelet om aktuella odds ar over breakeven. Ar de under — hoppa over, oavsett hur lockande spelet ser ut i ovrigt. Det ar den enda matematiskt korrekta gransen.
- Var arlig: traffchansen ar en uppskattning, inte ett facit. Saga det i texten.

Valj "Hoppa over" nar: marknaden ser effektiv ut, inget tydligt vardegap finns, faltet ar for otydligt, eller du maste stracja pa resonemang for att hitta ett spel. Att saga hoppa over ar ett lika bra svar som att rekommendera ett spel - ofta ett battre."""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=anthropic_headers(api_key),
                json={
                    "model": MODEL,
                    "max_tokens": 3500,
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
