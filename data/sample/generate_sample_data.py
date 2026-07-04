"""
Sample data generator for the FIFA 2026 AI Knockout + Sponsor Stock dashboard.

Produces three deterministic JSON files in this directory:

  teams.json    - per-team tournament statistics (group stage + Round of 32)
  players.json  - key player statistics per team
  matches.json  - full knockout bracket (R32 completed, R16 upcoming, later TBD)

The output format is the "contract" the rest of the app consumes. To plug in
real FIFA 2026 data later, replace these files (or point FifaDataService at a
real API) while keeping the same keys.

Run:  python generate_sample_data.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RNG = random.Random(2026)  # fixed seed -> deterministic sample data

# ---------------------------------------------------------------------------
# 32 knockout-stage teams. "tier" is a 0-100 base strength used only to
# synthesize plausible statistics; the prediction model never sees it.
# ---------------------------------------------------------------------------
TEAMS = {
    "Argentina":    {"code": "ARG", "flag": "\U0001F1E6\U0001F1F7", "tier": 95, "group": "A",
                     "players": [("Lionel Messi", "FW"), ("Julián Álvarez", "FW"), ("Enzo Fernández", "MF"), ("Emiliano Martínez", "GK")]},
    "France":       {"code": "FRA", "flag": "\U0001F1EB\U0001F1F7", "tier": 94, "group": "B",
                     "players": [("Kylian Mbappé", "FW"), ("Antoine Griezmann", "MF"), ("Aurélien Tchouaméni", "MF"), ("Mike Maignan", "GK")]},
    "Spain":        {"code": "ESP", "flag": "\U0001F1EA\U0001F1F8", "tier": 93, "group": "C",
                     "players": [("Lamine Yamal", "FW"), ("Pedri", "MF"), ("Rodri", "MF"), ("Unai Simón", "GK")]},
    "England":      {"code": "ENG", "flag": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F", "tier": 91, "group": "D",
                     "players": [("Harry Kane", "FW"), ("Jude Bellingham", "MF"), ("Bukayo Saka", "FW"), ("Jordan Pickford", "GK")]},
    "Brazil":       {"code": "BRA", "flag": "\U0001F1E7\U0001F1F7", "tier": 90, "group": "E",
                     "players": [("Vinícius Júnior", "FW"), ("Rodrygo", "FW"), ("Casemiro", "MF"), ("Alisson", "GK")]},
    "Portugal":     {"code": "POR", "flag": "\U0001F1F5\U0001F1F9", "tier": 89, "group": "F",
                     "players": [("Cristiano Ronaldo", "FW"), ("Bruno Fernandes", "MF"), ("Bernardo Silva", "MF"), ("Diogo Costa", "GK")]},
    "Netherlands":  {"code": "NED", "flag": "\U0001F1F3\U0001F1F1", "tier": 87, "group": "G",
                     "players": [("Cody Gakpo", "FW"), ("Memphis Depay", "FW"), ("Frenkie de Jong", "MF"), ("Virgil van Dijk", "DF")]},
    "Germany":      {"code": "GER", "flag": "\U0001F1E9\U0001F1EA", "tier": 86, "group": "H",
                     "players": [("Jamal Musiala", "MF"), ("Florian Wirtz", "MF"), ("Joshua Kimmich", "MF"), ("Marc-André ter Stegen", "GK")]},
    "Italy":        {"code": "ITA", "flag": "\U0001F1EE\U0001F1F9", "tier": 85, "group": "I",
                     "players": [("Nicolò Barella", "MF"), ("Mateo Retegui", "FW"), ("Federico Chiesa", "FW"), ("Gianluigi Donnarumma", "GK")]},
    "Belgium":      {"code": "BEL", "flag": "\U0001F1E7\U0001F1EA", "tier": 84, "group": "J",
                     "players": [("Kevin De Bruyne", "MF"), ("Romelu Lukaku", "FW"), ("Jérémy Doku", "FW"), ("Thibaut Courtois", "GK")]},
    "Croatia":      {"code": "CRO", "flag": "\U0001F1ED\U0001F1F7", "tier": 83, "group": "K",
                     "players": [("Luka Modrić", "MF"), ("Mateo Kovačić", "MF"), ("Andrej Kramarić", "FW"), ("Dominik Livaković", "GK")]},
    "Uruguay":      {"code": "URU", "flag": "\U0001F1FA\U0001F1FE", "tier": 83, "group": "L",
                     "players": [("Federico Valverde", "MF"), ("Darwin Núñez", "FW"), ("Manuel Ugarte", "MF"), ("Sergio Rochet", "GK")]},
    "Morocco":      {"code": "MAR", "flag": "\U0001F1F2\U0001F1E6", "tier": 82, "group": "B",
                     "players": [("Achraf Hakimi", "DF"), ("Youssef En-Nesyri", "FW"), ("Sofyan Amrabat", "MF"), ("Yassine Bounou", "GK")]},
    "Colombia":     {"code": "COL", "flag": "\U0001F1E8\U0001F1F4", "tier": 82, "group": "A",
                     "players": [("Luis Díaz", "FW"), ("James Rodríguez", "MF"), ("Richard Ríos", "MF"), ("Camilo Vargas", "GK")]},
    "USA":          {"code": "USA", "flag": "\U0001F1FA\U0001F1F8", "tier": 80, "group": "C",
                     "players": [("Christian Pulisic", "FW"), ("Folarin Balogun", "FW"), ("Weston McKennie", "MF"), ("Matt Turner", "GK")]},
    "Denmark":      {"code": "DEN", "flag": "\U0001F1E9\U0001F1F0", "tier": 79, "group": "D",
                     "players": [("Rasmus Højlund", "FW"), ("Christian Eriksen", "MF"), ("Pierre-Emile Højbjerg", "MF")]},
    "Mexico":       {"code": "MEX", "flag": "\U0001F1F2\U0001F1FD", "tier": 78, "group": "E",
                     "players": [("Santiago Giménez", "FW"), ("Edson Álvarez", "MF"), ("Hirving Lozano", "FW")]},
    "Switzerland":  {"code": "SUI", "flag": "\U0001F1E8\U0001F1ED", "tier": 78, "group": "F",
                     "players": [("Granit Xhaka", "MF"), ("Breel Embolo", "FW"), ("Yann Sommer", "GK")]},
    "Japan":        {"code": "JPN", "flag": "\U0001F1EF\U0001F1F5", "tier": 78, "group": "G",
                     "players": [("Takefusa Kubo", "FW"), ("Kaoru Mitoma", "FW"), ("Wataru Endo", "MF")]},
    "Norway":       {"code": "NOR", "flag": "\U0001F1F3\U0001F1F4", "tier": 77, "group": "H",
                     "players": [("Erling Haaland", "FW"), ("Martin Ødegaard", "MF"), ("Alexander Sørloth", "FW")]},
    "Austria":      {"code": "AUT", "flag": "\U0001F1E6\U0001F1F9", "tier": 76, "group": "I",
                     "players": [("Marcel Sabitzer", "MF"), ("Konrad Laimer", "MF"), ("Marko Arnautović", "FW")]},
    "Senegal":      {"code": "SEN", "flag": "\U0001F1F8\U0001F1F3", "tier": 76, "group": "J",
                     "players": [("Sadio Mané", "FW"), ("Ismaïla Sarr", "FW"), ("Kalidou Koulibaly", "DF")]},
    "Ecuador":      {"code": "ECU", "flag": "\U0001F1EA\U0001F1E8", "tier": 75, "group": "K",
                     "players": [("Moisés Caicedo", "MF"), ("Kendry Páez", "MF"), ("Piero Hincapié", "DF")]},
    "South Korea":  {"code": "KOR", "flag": "\U0001F1F0\U0001F1F7", "tier": 74, "group": "L",
                     "players": [("Son Heung-min", "FW"), ("Lee Kang-in", "MF"), ("Kim Min-jae", "DF")]},
    "Canada":       {"code": "CAN", "flag": "\U0001F1E8\U0001F1E6", "tier": 73, "group": "A",
                     "players": [("Alphonso Davies", "DF"), ("Jonathan David", "FW"), ("Stephen Eustáquio", "MF")]},
    "Australia":    {"code": "AUS", "flag": "\U0001F1E6\U0001F1FA", "tier": 72, "group": "B",
                     "players": [("Jackson Irvine", "MF"), ("Craig Goodwin", "FW"), ("Mathew Ryan", "GK")]},
    "Iran":         {"code": "IRN", "flag": "\U0001F1EE\U0001F1F7", "tier": 72, "group": "C",
                     "players": [("Mehdi Taremi", "FW"), ("Sardar Azmoun", "FW"), ("Alireza Jahanbakhsh", "MF")]},
    "Algeria":      {"code": "ALG", "flag": "\U0001F1E9\U0001F1FF", "tier": 71, "group": "D",
                     "players": [("Riyad Mahrez", "FW"), ("Ismaël Bennacer", "MF"), ("Mohamed Amoura", "FW")]},
    "Egypt":        {"code": "EGY", "flag": "\U0001F1EA\U0001F1EC", "tier": 71, "group": "E",
                     "players": [("Mohamed Salah", "FW"), ("Omar Marmoush", "FW"), ("Trézéguet", "MF")]},
    "Saudi Arabia": {"code": "KSA", "flag": "\U0001F1F8\U0001F1E6", "tier": 69, "group": "F",
                     "players": [("Salem Al-Dawsari", "FW"), ("Firas Al-Buraikan", "FW"), ("Mohammed Al-Owais", "GK")]},
    "Qatar":        {"code": "QAT", "flag": "\U0001F1F6\U0001F1E6", "tier": 68, "group": "G",
                     "players": [("Akram Afif", "FW"), ("Almoez Ali", "FW"), ("Hassan Al-Haydos", "MF")]},
    "Panama":       {"code": "PAN", "flag": "\U0001F1F5\U0001F1E6", "tier": 66, "group": "H",
                     "players": [("Adalberto Carrasquilla", "MF"), ("Ismael Díaz", "FW"), ("Michael Murillo", "DF")]},
}

# ---------------------------------------------------------------------------
# Knockout bracket.
# R32 matches are completed (scores below). R16 is scheduled; QF/SF/F are TBD
# and reference the matches that feed them.
# Format R32: (id, home, away, (hs, as), (pen_h, pen_a) or None, date, venue)
# ---------------------------------------------------------------------------
R32_RESULTS = [
    ("M1",  "Argentina",   "Panama",       (3, 1), None,   "2026-06-28", "MetLife Stadium, New York/New Jersey"),
    ("M2",  "Netherlands", "South Korea",  (2, 0), None,   "2026-06-28", "BC Place, Vancouver"),
    ("M3",  "Spain",       "Egypt",        (2, 0), None,   "2026-06-29", "Lincoln Financial Field, Philadelphia"),
    ("M4",  "Portugal",    "Switzerland",  (1, 0), None,   "2026-06-29", "Gillette Stadium, Boston"),
    ("M5",  "France",      "Saudi Arabia", (4, 1), None,   "2026-06-30", "NRG Stadium, Houston"),
    ("M6",  "Germany",     "Croatia",      (2, 2), (5, 4), "2026-06-30", "AT&T Stadium, Dallas"),
    ("M7",  "England",     "Ecuador",      (2, 0), None,   "2026-07-01", "Mercedes-Benz Stadium, Atlanta"),
    ("M8",  "Brazil",      "Algeria",      (3, 0), None,   "2026-07-01", "Hard Rock Stadium, Miami"),
    ("M9",  "USA",         "Norway",       (1, 0), None,   "2026-06-28", "SoFi Stadium, Los Angeles"),
    ("M10", "Morocco",     "Belgium",      (1, 0), None,   "2026-06-29", "Levi's Stadium, San Francisco Bay Area"),
    ("M11", "Uruguay",     "Japan",        (2, 1), None,   "2026-06-30", "Estadio BBVA, Monterrey"),
    ("M12", "Colombia",    "Senegal",      (1, 1), (4, 2), "2026-07-01", "Estadio Akron, Guadalajara"),
    ("M13", "Mexico",      "Austria",      (2, 1), None,   "2026-07-02", "Estadio Azteca, Mexico City"),
    ("M14", "Italy",       "Australia",    (1, 0), None,   "2026-07-02", "BMO Field, Toronto"),
    ("M15", "Denmark",     "Canada",       (2, 0), None,   "2026-07-03", "Lumen Field, Seattle"),
    ("M16", "Iran",        "Qatar",        (2, 1), None,   "2026-07-03", "Arrowhead Stadium, Kansas City"),
]

# (id, home_from, away_from, date, venue)
R16_SCHEDULE = [
    ("M17", "M1",  "M2",  "2026-07-04", "NRG Stadium, Houston"),
    ("M18", "M3",  "M4",  "2026-07-04", "AT&T Stadium, Dallas"),
    ("M19", "M5",  "M6",  "2026-07-05", "Mercedes-Benz Stadium, Atlanta"),
    ("M20", "M7",  "M8",  "2026-07-05", "MetLife Stadium, New York/New Jersey"),
    ("M21", "M9",  "M10", "2026-07-06", "SoFi Stadium, Los Angeles"),
    ("M22", "M11", "M12", "2026-07-06", "Arrowhead Stadium, Kansas City"),
    ("M23", "M13", "M14", "2026-07-07", "Estadio Azteca, Mexico City"),
    ("M24", "M15", "M16", "2026-07-07", "Lumen Field, Seattle"),
]
QF_SCHEDULE = [
    ("M25", "M17", "M18", "2026-07-09", "Gillette Stadium, Boston"),
    ("M26", "M19", "M20", "2026-07-10", "SoFi Stadium, Los Angeles"),
    ("M27", "M21", "M22", "2026-07-11", "Hard Rock Stadium, Miami"),
    ("M28", "M23", "M24", "2026-07-11", "Arrowhead Stadium, Kansas City"),
]
SF_SCHEDULE = [
    ("M29", "M25", "M26", "2026-07-14", "AT&T Stadium, Dallas"),
    ("M30", "M27", "M28", "2026-07-15", "Mercedes-Benz Stadium, Atlanta"),
]
FINAL_SCHEDULE = [
    ("M31", "M29", "M30", "2026-07-19", "MetLife Stadium, New York/New Jersey"),
]

# Notable knockout surprises, used by the stock model as an "upset score"
# signal (bigger = more global attention / market chatter).
UPSETS = [
    {"date": "2026-06-29", "description": "Morocco knock out Belgium 1-0", "magnitude": 0.80},
    {"date": "2026-06-30", "description": "Germany-Croatia decided on penalties", "magnitude": 0.35},
    {"date": "2026-07-01", "description": "Colombia-Senegal decided on penalties", "magnitude": 0.25},
    {"date": "2026-07-03", "description": "Denmark eliminate co-host Canada", "magnitude": 0.45},
]


def winner(home, away, score, pens):
    hs, as_ = score
    if hs > as_:
        return home
    if as_ > hs:
        return away
    return home if pens[0] > pens[1] else away


def group_points_for(tier):
    """Plausible group-stage points for a team that reached the knockouts."""
    if tier >= 88:
        return RNG.choice([7, 9, 9])
    if tier >= 80:
        return RNG.choice([6, 7, 7])
    if tier >= 74:
        return RNG.choice([5, 6, 6])
    return RNG.choice([3, 4, 4, 5])


def results_from_points(points):
    """Turn group points into an oldest-to-newest W/D/L sequence."""
    table = {9: "WWW", 7: "WWD", 6: "WWL", 5: "WDD", 4: "WDL", 3: "DDL"}
    seq = list(table[points])
    RNG.shuffle(seq)
    return seq


def build_team_stats():
    teams_out = {}
    for name, meta in TEAMS.items():
        t = meta["tier"]
        s = (t - 60) / 35.0                      # 0..1 strength scalar
        n = lambda a: RNG.uniform(-a, a)          # small noise helper

        points = group_points_for(t)
        results = results_from_points(points)

        gf = max(1, round(3 * (0.9 + 2.1 * s + n(0.35))))
        ga = max(0, round(3 * (1.75 - 1.35 * s + n(0.30))))
        possession = round(45 + 17 * s + n(3.0), 1)
        shots_pm = round(8.0 + 9.5 * s + n(1.2), 1)
        sot_pm = round(shots_pm * (0.30 + 0.12 * s) + n(0.4), 1)
        pass_acc = round(76 + 15 * s + n(2.0), 1)
        fouls_pm = round(13.5 - 4.0 * s + n(1.2), 1)
        tackles_pm = round(15.0 + 4.0 * (1 - s) + n(1.5), 1)
        interceptions_pm = round(8.0 + 2.5 * (1 - s) + n(1.2), 1)
        saves_pm = round(2.2 + 1.6 * (1 - s) + n(0.5), 1)

        matches = 3
        yellows = max(1, round(fouls_pm * matches * (0.16 + n(0.03))))
        reds = 1 if RNG.random() < (0.16 - 0.10 * s) else 0
        clean_sheets = min(matches, max(0, round(0.4 + 2.2 * s + n(0.6))))

        teams_out[name] = {
            "name": name, "code": meta["code"], "flag": meta["flag"], "group": meta["group"],
            "matches_played": matches,
            "wins": results.count("W"), "draws": results.count("D"), "losses": results.count("L"),
            "goals_for": gf, "goals_against": ga,
            "group_points": points,
            "possession": possession,
            "shots_per_match": shots_pm,
            "shots_on_target_per_match": sot_pm,
            "passing_accuracy": pass_acc,
            "fouls_per_match": fouls_pm,
            "yellow_cards": yellows, "red_cards": reds,
            "clean_sheets": clean_sheets,
            "tackles_per_match": tackles_pm,
            "interceptions_per_match": interceptions_pm,
            "saves_per_match": saves_pm,
            "recent_results": results,          # oldest -> newest
        }

    # Fold the completed Round-of-32 games into each team's totals.
    for mid, home, away, score, pens, date, venue in R32_RESULTS:
        hs, as_ = score
        win = winner(home, away, score, pens)
        for team, gf_, ga_ in ((home, hs, as_), (away, as_, hs)):
            st = teams_out[team]
            st["matches_played"] += 1
            st["goals_for"] += gf_
            st["goals_against"] += ga_
            if ga_ == 0:
                st["clean_sheets"] += 1
            # Penalty wins/losses count as W/L for form purposes.
            outcome = "W" if team == win else "L"
            if gf_ > ga_:
                st["wins"] += 1
            elif gf_ < ga_:
                st["losses"] += 1
            else:
                st["draws"] += 1
            st["recent_results"].append(outcome)

    return teams_out


def build_players(teams_stats):
    """Distribute each team's goals/assists across its listed key players."""
    goal_shares = [0.45, 0.28, 0.16, 0.06]
    assist_shares = [0.22, 0.34, 0.26, 0.10]
    players_out = []
    for name, meta in TEAMS.items():
        st = teams_stats[name]
        mp = st["matches_played"]
        for i, (pname, pos) in enumerate(meta["players"]):
            goals = round(st["goals_for"] * goal_shares[i]) if pos != "GK" else 0
            assists = round(st["goals_for"] * assist_shares[i]) if pos != "GK" else 0
            apps = mp if i < 2 else max(1, mp - RNG.choice([0, 0, 1]))
            minutes = apps * RNG.randint(72, 90)
            defensive = pos in ("DF", "MF", "GK")
            players_out.append({
                "team": name,
                "name": pname,
                "position": pos,
                "goals": goals,
                "assists": assists,
                "minutes": minutes,
                "appearances": apps,
                "yellow_cards": RNG.choice([0, 0, 0, 1, 1, 2]),
                "red_cards": 1 if RNG.random() < 0.02 else 0,
                "tackles": round((3.2 if defensive else 0.9) * apps + RNG.uniform(0, 2), 1),
                "interceptions": round((2.4 if defensive else 0.6) * apps + RNG.uniform(0, 2), 1),
                "saves": round(st["saves_per_match"] * apps, 1) if pos == "GK" else 0,
                "rating": round(min(9.4, 5.6 + (TEAMS[name]["tier"] - 60) / 14.0 + RNG.uniform(-0.4, 0.6) + (0.5 if i == 0 else 0)), 2),
            })
    return players_out


def build_matches():
    matches = []
    for mid, home, away, score, pens, date, venue in R32_RESULTS:
        matches.append({
            "id": mid, "round": "R32", "date": date, "venue": venue,
            "home": home, "away": away,
            "status": "completed",
            "home_score": score[0], "away_score": score[1],
            "penalties": {"home": pens[0], "away": pens[1]} if pens else None,
            "winner": winner(home, away, score, pens),
            "home_from": None, "away_from": None,
        })
    r32_winner = {m["id"]: m["winner"] for m in matches}

    for mid, hf, af, date, venue in R16_SCHEDULE:
        matches.append({
            "id": mid, "round": "R16", "date": date, "venue": venue,
            "home": r32_winner[hf], "away": r32_winner[af],
            "status": "upcoming",
            "home_score": None, "away_score": None, "penalties": None, "winner": None,
            "home_from": hf, "away_from": af,
        })
    for schedule, rnd in ((QF_SCHEDULE, "QF"), (SF_SCHEDULE, "SF"), (FINAL_SCHEDULE, "F")):
        for mid, hf, af, date, venue in schedule:
            matches.append({
                "id": mid, "round": rnd, "date": date, "venue": venue,
                "home": None, "away": None,          # participants not yet known
                "status": "scheduled",
                "home_score": None, "away_score": None, "penalties": None, "winner": None,
                "home_from": hf, "away_from": af,
            })
    return {
        "tournament": "FIFA World Cup 2026",
        "start_date": "2026-06-11",
        "rounds": ["R32", "R16", "QF", "SF", "F"],
        "matches": matches,
        "upsets": UPSETS,
        "note": "SAMPLE DATA - deterministic synthetic tournament state as of 2026-07-04.",
    }


def main():
    teams = build_team_stats()
    players = build_players(teams)
    matches = build_matches()

    (HERE / "teams.json").write_text(json.dumps(teams, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "players.json").write_text(json.dumps(players, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "matches.json").write_text(json.dumps(matches, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(teams)} teams, {len(players)} players, {len(matches['matches'])} matches -> {HERE}")


if __name__ == "__main__":
    main()
