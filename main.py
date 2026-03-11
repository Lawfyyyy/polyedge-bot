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

def is_market_active(m: dict) -> bool:
    """Vérifie qu'un marché est vraiment tradable (pas résolu, accepting orders)."""
    if m.get("closed") or not m.get("active"):
        return False
    if not m.get("accepting_orders"):
        return False
    tokens = m.get("tokens", [])
    if not tokens:
        return False
    for t in tokens:
        try:
            price = float(t.get("price", 0))
        except (TypeError, ValueError):
            return False
        # Prix à 0 ou 1 = marché déjà résolu
        if price <= 0.005 or price >= 0.995:
            return False
    return True

def parse_hours_left(m: dict):
    """Calcule les heures restantes avant expiration. Retourne None si expiré."""
    now = datetime.now(timezone.utc)
    end_date = m.get("end_date_iso") or m.get("end_date") or m.get("expiration")
    if not end_date:
        return None
    try:
        if end_date.endswith("Z"):
            end_date = end_date[:-1] + "+00:00"
        exp = datetime.fromisoformat(end_date)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        diff = (exp - now).total_seconds() / 3600
        return round(diff, 1) if diff > 0 else None
    except Exception:
        return None

async def fetch_active_markets(client: httpx.AsyncClient, max_pages: int = 8) -> list:
    """Utilise l'API Gamma de Polymarket — tri par date d'expiration croissante."""
    all_markets = []
    offset = 0
    limit = 100

    for _ in range(max_pages):
        try:
            res = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "limit": limit,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                    "order": "end_date_iso",
                    "ascending": "true",
                },
                timeout=20,
            )
            data = res.json()
        except Exception:
            break

        if isinstance(data, list):
            page_markets = data
        else:
            page_markets = data.get("data", data.get("markets", []))

        if not page_markets:
            break

        all_markets.extend(page_markets)
        offset += limit

    return all_markets

# ─── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PolyEdge Bot en ligne", "version": "2.0"}

@app.get("/health")
def health():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"status": "ok", "key_found": bool(key), "key_len": len(key)}


@app.get("/markets")
async def get_markets(limit: int = 50, hours: int = 72):
    """
    Retourne uniquement les marchés ACTIFS, non résolus,
    qui expirent dans la fenêtre `hours` demandée.
    Trie par urgence (plus urgent en premier).
    """
    async with httpx.AsyncClient(timeout=20) as client:
        all_markets = await fetch_active_markets(client, max_pages=6)

        filtered = []
        for m in all_markets:
            # Filtre strict : actif + prix entre 0.01 et 0.99
            if not is_market_active(m):
                continue
            # Heures restantes
            h = parse_hours_left(m)
            if h is None:
                continue
            # Fenêtre temporelle demandée
            if not (0 < h <= hours):
                continue

            m["hours_left"] = h
            filtered.append(m)

        # Tri : plus urgent en premier
        filtered.sort(key=lambda m: m.get("hours_left", 9999))

        return filtered[:limit]


@app.get("/markets/recommended")
async def get_recommended_markets():
    """
    Claude scanne jusqu'à 30 jours de marchés et sélectionne
    lui-même les plus intéressants à trader, sans contrainte de date.
    """
    async with httpx.AsyncClient(timeout=25) as client:
        all_markets = await fetch_active_markets(client, max_pages=8)

    # Garder seulement les marchés actifs (fenêtre large : 30 jours)
    candidates = []
    for m in all_markets:
        if not is_market_active(m):
            continue
        h = parse_hours_left(m)
        if h is None or h <= 0 or h > 720:  # max 30 jours
            continue
        m["hours_left"] = h
        candidates.append(m)

    if not candidates:
        return {"markets": [], "total_scanned": 0}

    # Mix : urgents + prix équilibrés (plus d'incertitude = plus d'edge potentiel)
    by_urgency = sorted(candidates, key=lambda m: m.get("hours_left", 9999))[:15]
    by_balance = sorted(
        candidates,
        key=lambda m: abs(float((m.get("tokens") or [{"price": 0.5}])[0].get("price", 0.5)) - 0.5)
    )[:15]

    pool = list({m["condition_id"]: m for m in by_urgency + by_balance}.values())[:30]

    # Demande à Claude de choisir les 8 meilleurs
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    ai_client = anthropic.Anthropic(api_key=key)

    markets_text = "\n".join([
        f"{i+1}. [{round(m.get('hours_left', 0))}h] {m.get('question', '?')} | "
        f"YES={float((m.get('tokens') or [{'price': 0}])[0].get('price', 0)):.2f}"
        for i, m in enumerate(pool)
    ])

    prompt = f"""Tu es un expert en marchés de prédiction Polymarket.

Voici {len(pool)} marchés actifs (format: [heures restantes] question | prix YES) :

{markets_text}

Sélectionne les 8 marchés les plus intéressants à trader MAINTENANT.

Critères :
- Prix YES entre 0.10 et 0.90 (vraie incertitude)
- Question avec un enjeu mesurable et vérifiable
- Potentiel d'edge (actualité, données disponibles, marché mal pricé)
- Bon ratio risque/récompense

Réponds UNIQUEMENT en JSON strict, sans texte avant ni après :
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
        # Nettoyer les backticks si présents
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        recommendations = result.get("recommended", [])
    except Exception as e:
        return {"markets": pool, "total_scanned": len(candidates), "error": str(e)}

    # Enrichir avec les données IA
    enriched = []
    for rec in recommendations:
        idx = rec.get("index", 1) - 1
        if 0 <= idx < len(pool):
            market = pool[idx].copy()
            market["_ai_signal"]    = rec.get("signal")
            market["_ai_confiance"] = rec.get("confiance")
            market["_ai_raison"]    = rec.get("raison")
            market["_ai_edge"]      = rec.get("edge")
            market["_recommended"]  = True
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
        model="claude-sonnet-4-6",
        max_tokens=120,
        messages=[{"role": "user", "content": f"Traduis cette question de marché Polymarket en français, en une phrase courte et naturelle. Juste la traduction, rien d'autre: {data.get('text', '')}"}]
    )
    return {"translation": msg.content[0].text.strip()}


@app.post("/analyze")
async def analyze(data: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    single_market = data.get("single_market")
    markets       = data.get("markets", [])[:10]
    weather       = data.get("weather_cities", [])

    if single_market:
        weather_ctx = ""
        if single_market.get("weather_context"):
            wc = single_market["weather_context"]
            weather_ctx = f"\nContexte météo {wc.get('city', '')}: {wc.get('avg_temp', '?')}°F · pluie {wc.get('rain_chance', '?')}%"

        prompt = f"""Tu es expert Polymarket. Analyse ce marché en français. Sois direct et actionnable.

Question: {single_market.get('question', 'N/A')}
Prix YES: {single_market.get('yes_price', 'N/A')}
Prix NO: {single_market.get('no_price', 'N/A')}
Expire dans: {single_market.get('hours_left', 'N/A')} heures{weather_ctx}

Réponds avec ce format exact, une ligne par item :
📊 SIGNAL: OUI / NON / ABSTENTION
💯 CONFIANCE: XX%
💡 RAISON: [2 phrases max]
⚠️ RISQUE: [1 risque principal]
🎯 EDGE: [En quoi ce marché est potentiellement mal pricé]"""

    else:
        markets_text = "\n".join([
            f"- {m.get('question', 'N/A')} | YES: {(m.get('tokens') or [{}])[0].get('price', '?')} | {m.get('hours_left', '?')}h"
            for m in markets
        ])
        weather_text = "\n".join([
            f"- {w['city']}: {w.get('avg_temp', '?')}°F · pluie {w.get('rain_chance', '?')}%"
            for w in weather
        ]) if weather else "Aucune donnée météo chargée"

        prompt = f"""Tu es expert en marchés de prédiction Polymarket. Analyse en français, sois direct.

MARCHÉS ACTIFS :
{markets_text}

DONNÉES MÉTÉO :
{weather_text}

Pour chaque marché :
1. Traduction française (1 ligne)
2. Signal OUI/NON/ABSTENTION + confiance %
3. Raison en 1 phrase
4. Edge potentiel (pourquoi c'est une opportunité)

Classe du meilleur edge au pire. Commence directement par le n°1."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"analysis": msg.content[0].text}
