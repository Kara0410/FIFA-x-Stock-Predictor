"""
FIFA World Cup 2026 sponsor catalogue.

exposure_score (0..1) is a hand-set proxy for how visible the brand is
around the tournament (boards, broadcast, activations). region + related
teams let the stock model weight regional relevance.
"""
from __future__ import annotations

SPONSORS = [
    {"ticker": "ADS.DE",  "name": "Adidas",          "category": "Kit supplier / FIFA partner",
     "region": "Europe",        "exposure_score": 0.95, "currency": "EUR",
     "related_teams": ["Argentina", "Spain", "Germany", "Italy", "Mexico", "Japan", "Belgium"]},
    {"ticker": "KO",      "name": "Coca-Cola",       "category": "FIFA partner",
     "region": "North America", "exposure_score": 0.90, "currency": "USD", "related_teams": []},
    {"ticker": "V",       "name": "Visa",            "category": "FIFA partner",
     "region": "North America", "exposure_score": 0.85, "currency": "USD", "related_teams": []},
    {"ticker": "BAC",     "name": "Bank of America", "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.70, "currency": "USD", "related_teams": ["USA"]},
    {"ticker": "MCD",     "name": "McDonald's",      "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.80, "currency": "USD", "related_teams": []},
    {"ticker": "PEP",     "name": "PepsiCo (Lay's)", "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.65, "currency": "USD", "related_teams": []},
    {"ticker": "VZ",      "name": "Verizon",         "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.60, "currency": "USD", "related_teams": ["USA"]},
    {"ticker": "HD",      "name": "Home Depot",      "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.55, "currency": "USD", "related_teams": ["USA"]},
    {"ticker": "VVV",     "name": "Valvoline",       "category": "Tournament sponsor",
     "region": "North America", "exposure_score": 0.40, "currency": "USD", "related_teams": []},
    {"ticker": "LNVGY",   "name": "Lenovo (ADR)",    "category": "FIFA partner",
     "region": "Asia",          "exposure_score": 0.60, "currency": "USD", "related_teams": []},
    {"ticker": "2222.SR", "name": "Saudi Aramco",    "category": "FIFA partner",
     "region": "Middle East",   "exposure_score": 0.75, "currency": "SAR",
     "related_teams": ["Saudi Arabia"]},
]

# Rough baseline prices used ONLY for generating demo data when the real
# download fails (roughly in the right ballpark for each ticker).
DEMO_BASE_PRICES = {
    "ADS.DE": 231.0, "KO": 63.0, "V": 290.0, "BAC": 40.0, "MCD": 265.0,
    "PEP": 168.0, "VZ": 41.0, "HD": 350.0, "VVV": 38.0, "LNVGY": 24.0,
    "2222.SR": 27.5,
}


class SponsorService:
    def list_sponsors(self) -> list[dict]:
        return SPONSORS

    def get(self, ticker: str) -> dict | None:
        return next((s for s in SPONSORS if s["ticker"].upper() == ticker.upper()), None)

    def exposure_for(self, ticker: str, teams_alive: set[str] | None = None) -> float:
        """Exposure score, boosted while brand-related teams are still alive."""
        sponsor = self.get(ticker)
        if not sponsor:
            return 0.3
        score = sponsor["exposure_score"]
        if teams_alive:
            alive_related = [t for t in sponsor["related_teams"] if t in teams_alive]
            score = min(1.0, score + 0.05 * len(alive_related))
        return score
