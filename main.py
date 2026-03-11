from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import anthropic
import os
from dotenv import load_dotenv

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

@app.get("/markets")
async def get_markets():
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "https://clob.polymarket.com/markets",
            params={"limit": 20, "active": True},
            timeout=10
        )
        return res.json()

@app.get("/noaa/{city_lat}/{city_lng}")
async def get_noaa(city_lat: float, city_lng: float):
    async with httpx.AsyncClient() as client:
        point = await client.get(
            f"https://api.weather.gov/points/{city_lat},{city_lng}",
            headers={"User-Agent": "PolyEdgeBot/1.0"},
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
    
    markets_text = "\n".join([
        f"- {m.get('question', 'N/A')} | Prix YES: {m.get('bestAsk', 'N/A')}"
        for m in data.get("markets", [])[:10]
    ])
    
    noaa_text = f"Température moyenne: {data.get('avg_temp', 'N/A')}°F | Pluie: {data.get('rain_chance', 'N/A')}%"
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Tu es un analyseur de marchés de prédiction.

Voici les marchés Polymarket ouverts :
{markets_text}

Données météo NOAA actuelles :
{noaa_text}

Pour chaque marché pertinent réponds avec :
- OUI ou NON
- Niveau de confiance de 0 à 100
- Raison en une phrase

Sois concis et factuel."""
        }]
    )
    
    return {"analysis": message.content[0].text}

@app.get("/health")
def health():
    return {"status": "ok"}
