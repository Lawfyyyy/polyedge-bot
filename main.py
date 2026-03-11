from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import anthropic
import json
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Helpers ───────────────────────────────────────────────

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

def parse_hours_left(end_date: str) -> float | None:
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

async def fetch_gamma_markets(client: httpx.AsyncClient, max_pages: int = 8) -> list:
    all_markets = []
    offset = 0
    for _ in range(max_pages):
        try:
            r = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": 100, "offset": offset, "active": "true", "closed": "false"},
                timeout=20,
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

# ─── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PolyEdge Bot en ligne", "version": "3.0"}

@app.get("/health")
def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"status": "ok", "key_found": bool(key), "key_len": len(key)}

@app.get("/debug")
async def debug():
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get("https://gamma-api.polymarket.com/markets?limit=3&active=true&closed=false")
            data = r.json()
            page = data if isinstance(data, list) else data.get("data", [])
            first = page[0] if page else {}
            end = first.get("endDate") or first.get("endDateIso", "")
            hours = parse_hours_left(end)
            return {
                "status": r.status_code,
                "count_returned": len(page),
                "first_question": first.get("question"),
                "active": first.get("active"),
                "closed": first.get("closed"),
                "acceptingOrders": first.get("acceptingOrders"),
                "endDate": end,
                "hours_left_calculated": hours,
                "outcomePrices": first.get("outcomePrices"),
                "yes_price": get_yes_price(first),
            }
        except Exception as e:
            return {"error": str(e)}

@app.get("/markets")
async def get_markets(limit: int = 50, hours: int = 720):
    async with httpx.AsyncClient(timeout=30) as client:
        all_markets = await fetch_gamma_markets(client, max_pages=6)

    result = []
    for m in all_markets:
        if not m.get("active") or m.get("closed"):
            continue
        if not m.get("acceptingOrders"):
            continue

        # Prix entre 0.02 et 0.98
        yes_price = get_yes_price(m)
        if yes_price <= 0.02 or yes_price >= 0.98:
            continue

        # Heures restantes
        end = m.get("endDate") or m.get("endDateIso", "")
        h = parse_hours_left(end)
        if h is None or h <= 0:
            continue
        if h > hours:
            continue

        m["hours_left"] = h
        m["yes_price"] = yes_price
        result.append(m)

    result.sort(key=lambda m: m["hours_left"])
    return result[:limit]

@app.get("/markets/recommended")
async def get_recommended_markets():
    async with httpx.AsyncClient(timeout=30) as client:
        all_markets = await fetch_gamma_markets(client, max_pages=8)

    candidates = []
    for m in all_markets:
        if not m.get("active") or m.get("closed"):
            continue
        if not m.get("acceptingOrders"):
            continue
        yes_price = get_yes_price(m)
        if yes_price <= 0.05 or yes_price >= 0.95:
            continue
        end = m.get("endDate") or m.get("endDateIso", "")
        h = parse_hours_left(end)
        if h is None or h <= 0 or h > 720:
            continue
        m["hours_left"] = h
        m["yes_price"] = yes_price
        candidates.append(m)

    if not candidates:
        return {"markets": [], "total_scanned": len(all_markets)}

    by_urgency = sorted(candidates, key=lambda m: m["hours_left"])[:15]
    by_balance = sorted(candidates, key=lambda m: abs(m["yes_price"] - 0.5))[:15]
    pool = list({m["id"]: m for m in by_urgency + by_balance}.values())[:30]

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    ai_client = anthropic.Anthropic(api_key=key)

    markets_text = "\n".join([
        f"{i+1}. [{round(m['hours_left'])}h] {m.get('question','?')} | YES={m['yes_price']:.2f}"
        for i, m in enumerate(pool)
    ])

    prompt = f"""Tu es un expert en marchés de prédiction Polymarket.

Voici {len(pool)} marchés actifs (format: [heures restantes] question | prix YES) :

{markets_text}

Sélectionne les 8 marchés les plus intéressants à trader MAINTENANT.
Critères : vraie incertitude (prix 0.10-0.90), enjeu mesurable, potentiel d'edge.

Réponds UNIQUEMENT en JSON strict :
{{
  "recommended": [
    {{
      "index": <numéro 1-{len(pool)}>,
      "signal": "OUI" ou "NON",
      "confiance": <50-95>,
      "raison": "<1 phrase directe en français>",
      "edge": "<pourquoi ce marché est potentiellement mal pricé>"
    }}
  ]
}}"""

    try:
        msg = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        recommendations = result.get("recommended", [])
    except Exception as e:
        return {"markets": pool, "total_scanned": len(candidates), "error": str(e)}

    enriched = []
    for rec in recommendations:
        idx = rec.get("index", 1) - 1
        if 0 <= idx < len(pool):
            market = pool[idx].copy()
            market["_ai_signal"] = rec.get("signal")
            market["_ai_confiance"] = rec.get("confiance")
            market["_ai_raison"] = rec.get("raison")
            market["_ai_edge"] = rec.get("edge")
            market["_recommended"] = True
            enriched.append(market)

    return {"markets": enriched, "total_scanned": len(candidates)}

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

    if single_market:
        weather_ctx = ""
        if single_market.get("weather_context"):
            wc = single_market["weather_context"]
            weather_ctx = f"\nContexte météo {wc.get('city','')}: {wc.get('avg_temp','?')}°F · pluie {wc.get('rain_chance','?')}%"

        prompt = f"""Tu es expert Polymarket. Analyse ce marché en français. Sois direct et actionnable.

Question: {single_market.get('question','N/A')}
Prix YES: {single_market.get('yes_price','N/A')}
Prix NO: {single_market.get('no_price','N/A')}
Expire dans: {single_market.get('hours_left','N/A')} heures{weather_ctx}

Réponds avec ce format exact :
📊 SIGNAL: OUI / NON / ABSTENTION
💯 CONFIANCE: XX%
💡 RAISON: [2 phrases max]
⚠️ RISQUE: [1 risque principal]
🎯 EDGE: [En quoi ce marché est potentiellement mal pricé]"""

    else:
        markets_text = "\n".join([
            f"- {m.get('question','N/A')} | YES: {m.get('yes_price', get_yes_price(m))} | {m.get('hours_left','?')}h"
            for m in markets
        ])
        weather_text = "\n".join([
            f"- {w['city']}: {w.get('avg_temp','?')}°F · pluie {w.get('rain_chance','?')}%"
            for w in weather
        ]) if weather else "Aucune donnée météo"

        prompt = f"""Tu es expert en marchés de prédiction Polymarket. Analyse en français, sois direct.

MARCHÉS ACTIFS :
{markets_text}

DONNÉES MÉTÉO :
{weather_text}

Pour chaque marché : signal OUI/NON/ABSTENTION + confiance % + raison 1 phrase + edge potentiel.
Classe du meilleur edge au pire."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"analysis": msg.content[0].text}
