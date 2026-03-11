from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import anthropic
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "PolyEdge Bot en ligne"}

@app.get("/health")
def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"status": "ok", "key_found": bool(key), "key_len": len(key)}

@app.get("/markets")
async def get_markets(limit: int = 50, hours: int = 72):
    async with httpx.AsyncClient() as client:
        res = await client.get("https://clob.polymarket.com/markets", params={"limit": 100, "active": True}, timeout=15)
        data = res.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        now = datetime.now(timezone.utc)
        filtered = []
        for m in markets:
            end_date = m.get("end_date_iso") or m.get("end_date") or m.get("expiration")
            if end_date:
                try:
                    if end_date.endswith("Z"):
                        end_date = end_date[:-1] + "+00:00"
                    exp = datetime.fromisoformat(end_date)
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    diff_hours = (exp - now).total_seconds() / 3600
                    if 0 < diff_hours <= hours:
                        m["hours_left"] = round(diff_hours, 1)
                        filtered.append(m)
                except:
                    pass
        if len(filtered) == 0:
            filtered = markets[:limit]
            for m in filtered:
                m["hours_left"] = None
        return filtered[:limit]

@app.get("/noaa/{city_lat}/{city_lng}")
async def get_noaa(city_lat: float, city_lng: float):
    async with httpx.AsyncClient() as client:
        point = await client.get(f"https://api.weather.gov/points/{city_lat},{city_lng}", headers={"User-Agent": "PolyEdgeBot/1.0"}, timeout=10)
        data = point.json()
        forecast = await client.get(data["properties"]["forecastHourly"], timeout=10)
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
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Traduis cette question de marché en français en une phrase courte: {data.get('text','')}"}]
    )
    return {"translation": msg.content[0].text}

@app.post("/analyze")
async def analyze(data: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    single_market = data.get("single_market")
    markets = data.get("markets", [])[:10]
    weather = data.get("weather_cities", [])
    if single_market:
        prompt = f"""Analyse ce marché Polymarket en français :
Question: {single_market.get('question', 'N/A')}
Prix YES: {single_market.get('yes_price', 'N/A')}
Prix NO: {single_market.get('no_price', 'N/A')}
Expire dans: {single_market.get('hours_left', 'N/A')} heures

Réponds avec ce format exact :
🇫🇷 TRADUCTION: [traduction française]
📊 SIGNAL: OUI ou NON
💯 CONFIANCE: [0-100]%
💡 RAISON: [2 phrases max]
⚠️ RISQUE: [1 risque principal]"""
    else:
        markets_text = "\n".join([f"- {m.get('question','N/A')} | YES: {m.get('yes_price','?')} | {m.get('hours_left','?')}h" for m in markets])
        weather_text = "\n".join([f"- {w['city']}: {w.get('avg_temp','?')}°F · pluie {w.get('rain_chance','?')}%" for w in weather]) if weather else "Pas de données"
        prompt = f"""Tu es expert en marchés de prédiction Polymarket.
MARCHÉS (expirent dans 72h) :
{markets_text}
MÉTÉO NOAA :
{weather_text}
Pour chaque marché en français : traduction, signal OUI/NON, confiance %, raison courte. Commence par les meilleurs edges."""
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"analysis": msg.content[0].text}
