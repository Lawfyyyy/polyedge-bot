from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import anthropic
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_API_KEY")

@app.get("/")
def root():
    return {"status": "PolyEdge Bot en ligne"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/markets")
async def get_markets(limit: int = 50, hours: int = 72):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "https://clob.polymarket.com/markets",
            params={"limit": 100, "active": True},
            timeout=15
        )
        data = res.json()

        markets = []
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get("data", data.get("markets", []))

        # Filter: only markets expiring within next `hours` hours
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

        # If no markets in 72h window, return all active markets
        if len(filtered) == 0:
            filtered = markets[:limit]
            for m in filtered:
                m["hours_left"] = None

        return filtered[:limit]

@app.get("/noaa/{city_lat}/{city_lng}")
async def get_noaa(city_lat: float, city_lng: float):
    async with httpx.AsyncClient() as client:
        point = await client.get(
            f"https://api.weather.gov/points/{city_lat},{city_lng}",
            headers={"User-Agent": "PolyEdgeBot/1.0 (educational)"},
            timeout=10
        )
        data = point.json()
        forecast_url = data["properties"]["forecastHourly"]
        forecast = await client.get(forecast_url, timeout=10)
        periods = forecast.json()["properties"]["periods"][:6]
        return {
            "periods": periods,
            "rain_chance": max(p["probabilityOfPrecipitation"]["value"] or 0 for p in periods),
            "avg_temp": sum(p["temperature"] for p in periods) / len(periods)
        }

@app.post("/analyze")
async def analyze(data: dict):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    markets = data.get("markets", [])[:10]
    single_market = data.get("single_market")
    weather = data.get("weather_cities", [])

    if single_market:
        # Single market analysis with translation
        prompt = f"""Tu es un expert en marchés de prédiction. Analyse ce marché Polymarket :

Question originale (EN): {single_market.get('question', 'N/A')}
Prix YES actuel: {single_market.get('yes_price', 'N/A')}
Prix NO actuel: {single_market.get('no_price', 'N/A')}
Expire dans: {single_market.get('hours_left', 'N/A')} heures

Réponds en français avec exactement ce format :
🇫🇷 TRADUCTION: [traduction française de la question]
📊 SIGNAL: OUI ou NON
💯 CONFIANCE: [0-100]%
💡 RAISON: [explication en 2 phrases maximum]
⚠️ RISQUE: [un risque principal]"""
    else:
        # Full analysis
        markets_text = "\n".join([
            f"- {m.get('question', 'N/A')} | YES: {m.get('yes_price','?')} | Expire: {m.get('hours_left','?')}h"
            for m in markets
        ])
        weather_text = "\n".join([
            f"- {w['city']}: {w.get('avg_temp','?')}°F · pluie {w.get('rain_chance','?')}%"
            for w in weather
        ]) if weather else "Pas de données météo"

        prompt = f"""Tu es un expert en marchés de prédiction Polymarket.

MARCHÉS ACTIFS (expirent dans les 72h) :
{markets_text}

DONNÉES MÉTÉO NOAA :
{weather_text}

Pour chaque marché donne en français :
1. 🇫🇷 Traduction de la question
2. 📊 Signal: OUI / NON / ABSTENTION
3. 💯 Confiance: X%
4. 💡 Raison courte

Commence par les marchés avec le meilleur edge potentiel. Sois concis."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"analysis": message.content[0].text}

@app.post("/translate")
async def translate(data: dict):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    text = data.get("text", "")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Traduis cette question de marché de prédiction en français en une phrase courte: {text}"
        }]
    )
    return {"translation": message.content[0].text}
