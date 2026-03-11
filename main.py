from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import anthropic
import json
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_yes_price(m: dict) -> float:
    prices_raw = m.get("outcomePrices")
    if prices_raw:
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            return float(prices[0])
        except Exception:
            pass
    tokens = m.get("tokens") or []
    if tokens:
        try:
            return float(tokens[0].get("price", 0.5))
        except Exception:
            pass
    return 0.5

def parse_hours_left(end_date: str):
    if not end_date:
        return None
    now = datetime.now(timezone.utc)
    try:
        s = end_date.strip()
        if len(s) == 10:
            s = s + "T23:59:59+00:00"
        elif s.endswith("Z"):
            s = s[:-1] + "+00:00"
        exp = datetime.fromisoformat(s)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        diff = (exp - now).total_seconds() / 3600
        return round(diff, 1) if diff > 0 else None
    except Exception:
        return None

async def fetch_gamma_markets(client: httpx.AsyncClient, max_pages: int = 20) -> list:
    """Scanne jusqu'à 2000 marchés Gamma sans filtre temporel."""
    all_markets = []
    offset = 0
    for _ in range(max_pages):
        try:
            r = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "limit": 100,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
                timeout=25,
            )
            data = r.json()
        except Exception:
            break
        page = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        if not page:
            break
        all_markets.extend(page)
        offset += 100
    return all_markets

@app.get("/")
def root():
    return {"status": "PolyEdge Bot en ligne", "version": "4.0"}

@app.get("/health")
def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"status": "ok", "key_found": bool(key), "key_len": len(key)}

@app.get("/markets")
async def get_markets(limit: int = 200, hours: int = 8760):
    """
    Retourne tous les marchés actifs — pas de limite temporelle stricte.
    hours = filtre optionnel (défaut 1 an = tout garder)
    """
    async with httpx.AsyncClient(timeout=60) as client:
        all_markets = await fetch_gamma_markets(client, max_pages=20)

    result = []
    for m in all_markets:
        if not m.get("active") or m.get("closed"):
            continue
        if not m.get("acceptingOrders"):
            continue
        yes_price = get_yes_price(m)
        # Filtre souple: exclure seulement les marchés déjà résolus (0 ou 1)
        if yes_price <= 0.01 or yes_price >= 0.99:
            continue
        end = m.get("endDate") or m.get("endDateIso", "")
        h = parse_hours_left(end)
        # Garder même sans date connue
        m["hours_left"] = h
        m["yes_price"] = yes_price
        result.append(m)

    # Tri: marchés avec date d'abord (urgents en tête), puis sans date
    result.sort(key=lambda m: m.get("hours_left") or 99999)
    return result[:limit]

@app.get("/debug")
async def debug():
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get("https://gamma-api.polymarket.com/markets?limit=5&active=true&closed=false")
            data = r.json()
            page = data if isinstance(data, list) else data.get("data", [])
            return {
                "status": r.status_code,
                "count": len(page),
                "samples": [{"q": m.get("question","")[:60], "yp": get_yes_price(m), "h": parse_hours_left(m.get("endDate") or m.get("endDateIso",""))} for m in page[:3]]
            }
        except Exception as e:
            return {"error": str(e)}

@app.get("/noaa/{city_lat}/{city_lng}")
async def get_noaa(city_lat: float, city_lng: float):
    async with httpx.AsyncClient(timeout=15) as client:
        point = await client.get(
            f"https://api.weather.gov/points/{city_lat},{city_lng}",
            headers={"User-Agent": "PolyEdgeBot/1.0"}
        )
        data = point.json()
        forecast_url = data["properties"]["forecastHourly"]
        forecast = await client.get(forecast_url, timeout=15)
        periods = forecast.json()["properties"]["periods"][:6]
        return {
            "periods": periods,
            "rain_chance": max(p["probabilityOfPrecipitation"]["value"] or 0 for p in periods),
            "avg_temp": sum(p["temperature"] for p in periods) / len(periods)
        }

@app.post("/translate")
async def translate(data: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": f"Traduis cette question de marché Polymarket en français, en une phrase courte et naturelle. Juste la traduction, rien d'autre: {data.get('text', '')}"}]
    )
    return {"translation": msg.content[0].text.strip()}

@app.post("/analyze")
async def analyze(data: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    single_market = data.get("single_market")
    markets = data.get("markets", [])[:10]
    weather = data.get("weather_cities", [])
    news_search = data.get("news_search", False)

    if news_search:
        query = data.get("query", "polymarket trending")
        mkt_ctx = "\n".join([f"- {m.get('question','')[:80]} | YES={m.get('yes_price','?')}" for m in data.get("markets", [])[:8]])
        prompt = f"""Tu es un analyste de marchés de prédiction. Fais une revue d'actualités pertinentes pour ces marchés Polymarket ouverts.

MARCHÉS OUVERTS :
{mkt_ctx}

REQUÊTE : {query}

Génère 8-12 actualités récentes et pertinentes (réelles ou très probables étant donné les marchés ouverts).
Pour chaque actualité, indique :
- La source (Reuters, AP, BBC, Bloomberg, ESPN, etc.)
- La date approximative (aujourd'hui = 11 mars 2026)
- Le titre accrocheur
- 2 phrases de contexte
- Impact sur les marchés (FORT/MODÉRÉ/FAIBLE)
- Quel(s) marché(s) sont concernés

Format pour chaque item :
SOURCE: [nom] | DATE: [date] | IMPACT: [niveau]
TITRE: [titre]
CONTEXTE: [2 phrases]
MARCHÉS: [marchés liés]
---"""

    elif single_market:
        weather_ctx = ""
        if single_market.get("weather_context"):
            wc = single_market["weather_context"]
            weather_ctx = f"\nMétéo {wc.get('city','')}: {wc.get('avg_temp','?')}°F · pluie {wc.get('rain_chance','?')}%"

        prompt = f"""Tu es expert Polymarket. Analyse ce marché. Sois direct et actionnable.

Question: {single_market.get('question','N/A')}
Prix YES: {single_market.get('yes_price','N/A')} | Prix NO: {single_market.get('no_price','N/A')}
Expire dans: {single_market.get('hours_left','N/A')} heures
Volume 24h: ${single_market.get('volume_24h',0):,.0f} | Liquidité: ${single_market.get('liquidity',0):,.0f}{weather_ctx}

Format EXACT (une ligne par item) :
📊 SIGNAL: OUI / NON / ABSTENTION
💯 CONFIANCE: XX%
💡 RAISON: [2 phrases max, factuel]
⚠️ RISQUE: [1 risque principal]
🎯 EDGE: [écart estimé entre prix affiché et probabilité réelle, ex: +15pts]"""

    else:
        markets_text = "\n".join([
            f"{i+1}. {m.get('question','N/A')} | YES:{(m.get('tokens') or [{}])[0].get('price', m.get('yes_price','?'))} | {m.get('hours_left','?')}h"
            for i, m in enumerate(markets)
        ])
        weather_text = "\n".join([
            f"- {w['city']}: {w.get('avg_temp','?')}°F · pluie {w.get('rain_chance','?')}%"
            for w in weather
        ]) if weather else "Aucune donnée météo"

        prompt = f"""Tu es expert en marchés de prédiction Polymarket. Analyse ces marchés en français.

MARCHÉS :
{markets_text}

MÉTÉO :
{weather_text}

Pour chaque marché (du meilleur edge au pire) :
📊 SIGNAL: OUI/NON/ABSTENTION
💯 CONFIANCE: XX%
💡 RAISON: [1 phrase]
🎯 EDGE: [+/-Xpts vs prix affiché]"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"analysis": msg.content[0].text}
