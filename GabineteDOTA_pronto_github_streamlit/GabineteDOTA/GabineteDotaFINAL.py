# -*- coding: utf-8 -*-
"""
GabineteDotaFINAL.py
--------------------
Aplicação Tkinter no estilo do "GabineteFINAL3.0" (LoL), adaptada para Dota.

Objetivo (Dota v1 COMPLETO para linhas principais):
- Selecionar Liga (ou Todas), Time 1 e Time 2
- Abrir janela "Adicionar/Mudar Linhas" (aceita múltiplas linhas por campo)
- Calcular:
  1) Totais (Over/Under) de: Kills, Torres, Roshans, Barracks, Tempo
  2) Handicaps (T1) de: Kills, Torres, Roshans, Barracks
- Exibir resultados no mesmo "jeito" do Gabinete: tabela de análise por período:
  "Ano Todo", "10 jogos", "5 jogos" e também H2H.

CSV esperado na MESMA PASTA deste .py, nome: data.csv (editável em CSV_NAME).
Formato esperado (mínimo):
  date, league,
  teamname, opp_teamname,
  teamkills, oppkills,
  towers, opp_towers,
  roshans, opp_roshans,
  barracks, opp_barracks,
  duration  (segundos)  [opcional; se não existir, Tempo fica indisponível]

Cada linha do CSV deve ser "1 time em 1 partida".
"""

from __future__ import annotations

import os
import re
import math
import unicodedata
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Tkinter é usado apenas na interface desktop. Para a versão web, use streamlit_app.py.
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
except Exception:  # Tkinter pode não existir em ambientes Linux/headless (ex.: Streamlit Cloud)
    tk = None
    ttk = None
    messagebox = None
    ScrolledText = None

# Em ambientes sem Tkinter (ex.: Streamlit Cloud), evite acessar atributos de `tk` em tempo de import.
# Isso permite importar este módulo e reutilizar a parte de lógica/análise (dataset/analyzer/linhas) sem GUI.
_TK_AVAILABLE = tk is not None
_ToplevelBase = tk.Toplevel if _TK_AVAILABLE else object
# ==========================
# CONFIG
# ==========================

CSV_NAME = "data.csv"
SAVED_GAMES_FILE = "saved_games_dota.json"


# ==========================
# Utils (strings, parsing)
# ==========================

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def norm_team(s: str) -> str:
    s0 = _strip_accents(str(s or "")).lower().strip()
    for suf in [" esports", " e-sports", " esport", " gaming", " team", " club"]:
        if s0.endswith(suf):
            s0 = s0[: -len(suf)]
    s0 = s0.replace(".", " ").replace("_", " ")
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0


def parse_float_list(text: str) -> List[float]:
    """
    Aceita: "-2.5, -1.5 ; 0.5" etc.
    Ignora inválidos silenciosamente.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: List[float] = []
    for part in text.replace(";", ",").split(","):
        p = part.strip().replace(" ", "")
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def parse_time_line(text: str) -> List[float]:
    """
    Aceita várias linhas; cada uma pode ser:
    - "38.5" (minutos)
    - "38:30" (MM:SS)
    Retorna lista em minutos (float)
    """
    vals: List[float] = []
    for part in (text or "").replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        if ":" in p:
            try:
                mm, ss = p.split(":")
                mm = int(mm.strip())
                ss = int(ss.strip())
                vals.append(mm + ss / 60.0)
            except Exception:
                continue
        else:
            try:
                vals.append(float(p))
            except ValueError:
                continue
    return vals


def fmt_prob_and_odds(wins: int, total: int, pushes: int = 0, laplace: bool = True) -> Tuple[str, float]:
    """
    Formata no padrão "wins/valid (p%) | Odd: X.XXx".
    - Se pushes > 0, valid = total - pushes (push não entra em over/under).
    - Laplace: (wins+1)/(valid+2) para estabilizar amostras pequenas.
    Retorna (texto, odd_justa). Se valid==0 -> ("—", NaN).
    """
    valid = max(0, int(total) - int(pushes))
    if valid <= 0:
        return "—", float("nan")

    w = max(0, int(wins))
    if laplace:
        p = (w + 1) / (valid + 2)
    else:
        p = w / valid if valid else 0.0

    # evita divisão por zero
    if p <= 0:
        odd = float("inf")
    else:
        odd = 1.0 / p

    txt = f"{w}/{valid} ({p*100:.1f}%) | Odd: {odd:.2f}x"
    return txt, odd

def fmt_mmss_from_min(minutes: float) -> str:
    """
    Converte minutos (float) em string MM:SS.
    Retorna "-" se minutes for NaN/None.
    """
    try:
        if minutes is None:
            return "-"
        if isinstance(minutes, float) and (minutes != minutes):  # NaN
            return "-"
        total_sec = int(round(float(minutes) * 60))
        mm = total_sec // 60
        ss = total_sec % 60
        return f"{mm:02d}:{ss:02d}"
    except Exception:
        return "-"


def fmt_min_to_mmss(minutes: float) -> str:
    try:
        if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
            return "—"
        m = float(minutes)
        if m < 0:
            return "—"
        mm = int(m)
        ss = int(round((m - mm) * 60))
        if ss == 60:
            mm += 1
            ss = 0
        return f"{mm:02d}:{ss:02d}"
    except Exception:
        return "—"

    if laplace:
        p = (wins + 1) / (valid + 2)
    else:
        p = wins / valid

    odd = (1.0 / p) if p > 0 else float("inf")
    return f"{wins}/{valid} -> Odd: {odd:.2f}x", odd


# ==========================
# Core Data Model
# ==========================

@dataclass
class LinesConfig:
    # Totais O/U (linhas)
    total_kills: List[float] = field(default_factory=list)
    total_towers: List[float] = field(default_factory=list)
    total_roshans: List[float] = field(default_factory=list)
    total_barracks: List[float] = field(default_factory=list)
    total_time_min: List[float] = field(default_factory=list)  # minutos

    # Handicaps (para T1)
    hc_kills_t1: List[float] = field(default_factory=list)
    hc_towers_t1: List[float] = field(default_factory=list)
    hc_roshans_t1: List[float] = field(default_factory=list)
    hc_barracks_t1: List[float] = field(default_factory=list)


    def to_dict(self) -> dict:
        return {
            "total_kills": list(self.total_kills),
            "total_towers": list(self.total_towers),
            "total_roshans": list(self.total_roshans),
            "total_barracks": list(self.total_barracks),
            "total_time_min": list(self.total_time_min),
            "hc_kills_t1": list(self.hc_kills_t1),
            "hc_towers_t1": list(self.hc_towers_t1),
            "hc_roshans_t1": list(self.hc_roshans_t1),
            "hc_barracks_t1": list(self.hc_barracks_t1),
        }

    @staticmethod
    def from_dict(d: dict) -> "LinesConfig":
        d = d or {}
        return LinesConfig(
            total_kills=[float(x) for x in d.get("total_kills", [])],
            total_towers=[float(x) for x in d.get("total_towers", [])],
            total_roshans=[float(x) for x in d.get("total_roshans", [])],
            total_barracks=[float(x) for x in d.get("total_barracks", [])],
            total_time_min=[float(x) for x in d.get("total_time_min", [])],
            hc_kills_t1=[float(x) for x in d.get("hc_kills_t1", [])],
            hc_towers_t1=[float(x) for x in d.get("hc_towers_t1", [])],
            hc_roshans_t1=[float(x) for x in d.get("hc_roshans_t1", [])],
            hc_barracks_t1=[float(x) for x in d.get("hc_barracks_t1", [])],
        )

    def is_empty(self) -> bool:
        return not any([
            self.total_kills, self.total_towers, self.total_roshans, self.total_barracks, self.total_time_min,
            self.hc_kills_t1, self.hc_towers_t1, self.hc_roshans_t1, self.hc_barracks_t1
        ])


class DotaDataset:
    """
    Lê o CSV "por time" e fornece visões match-level e h2h pair-level.
    """

    def __init__(self, csv_name: str = CSV_NAME):
        base_dir = os.path.dirname(__file__)
        self.csv_path = os.path.join(base_dir, csv_name)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Arquivo '{csv_name}' não encontrado na mesma pasta:\n{base_dir}"
            )

        df = pd.read_csv(self.csv_path, low_memory=False)

        # Colunas mínimas
        required = [
            "date", "league",
            "teamname", "opp_teamname",
            "teamkills", "oppkills",
            "towers", "opp_towers",
            "roshans", "opp_roshans",
            "barracks", "opp_barracks"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV está faltando colunas obrigatórias: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["league"] = df["league"].fillna("Unknown League").astype(str)

        df["teamname"] = df["teamname"].astype(str).str.strip()
        df["opp_teamname"] = df["opp_teamname"].astype(str).str.strip()

        df["teamname_norm"] = df["teamname"].apply(norm_team)
        df["opp_teamname_norm"] = df["opp_teamname"].apply(norm_team)

        num_cols = [
            "teamkills", "oppkills",
            "towers", "opp_towers",
            "roshans", "opp_roshans",
            "barracks", "opp_barracks"
        ]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

        # duration opcional
        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        else:
            df["duration"] = np.nan

        # match_id opcional (melhor), senão cria chave a partir de (date, team, opp)
        if "match_id" in df.columns:
            df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
        else:
            # fallback: chave aproximada (não perfeito)
            df["match_id"] = (
                df["date"].astype(str) + "||" + df["teamname_norm"] + "||" + df["opp_teamname_norm"]
            )

        self.df = df

    def leagues(self) -> List[str]:
        ligas = sorted(self.df["league"].dropna().unique().tolist())
        return ["(Todas)"] + ligas

    def teams_for_league(self, league: str) -> List[str]:
        if league and league != "(Todas)":
            df = self.df[self.df["league"] == league]
        else:
            df = self.df

        teams = pd.unique(pd.concat([df["teamname"], df["opp_teamname"]], ignore_index=True))
        teams = [t for t in teams if isinstance(t, str) and t.strip()]
        return sorted(teams)

    def filter_league(self, league: str) -> pd.DataFrame:
        if league and league != "(Todas)":
            return self.df[self.df["league"] == league].copy()
        return self.df.copy()

    # --------------- Match-level totals for a given team (League sample) ---------------

    def match_totals_for_team(self, df_scope: pd.DataFrame, team_name: str) -> pd.DataFrame:
        """
        Retorna um DF com 1 linha por match (para os jogos do time),
        com colunas de totais do match: totalkills, totaltowers, totalroshans, totalbarracks, time_min
        """
        tn = norm_team(team_name)
        df_t = df_scope[df_scope["teamname_norm"] == tn].copy()
        if df_t.empty:
            return df_t

        # totais do match vindos da linha do time: stat_time + stat_opp
        df_t["totalkills"] = df_t["teamkills"] + df_t["oppkills"]
        df_t["totaltowers"] = df_t["towers"] + df_t["opp_towers"]
        df_t["totalroshans"] = df_t["roshans"] + df_t["opp_roshans"]
        df_t["totalbarracks"] = df_t["barracks"] + df_t["opp_barracks"]

        # tempo
        if "duration" in df_t.columns:
            df_t["time_min"] = df_t["duration"] / 60.0
        else:
            df_t["time_min"] = np.nan

        # garantir 1 linha por match_id (caso tenha duplicatas)
        df_t = df_t.sort_values("date", ascending=False).drop_duplicates(subset=["match_id"])
        return df_t


    def match_totals_all_matches(self, df_scope: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna DF com 1 linha por match_id (match-level) para o escopo,
        com totais: totalkills, totaltowers, totalroshans, totalbarracks, time_min.
        Usa uma das linhas do match (por time) e deduplica match_id.
        """
        if df_scope is None or df_scope.empty:
            return pd.DataFrame()

        dfm = df_scope.copy()
        dfm["totalkills"] = dfm["teamkills"] + dfm["oppkills"]
        dfm["totaltowers"] = dfm["towers"] + dfm["opp_towers"]
        dfm["totalroshans"] = dfm["roshans"] + dfm["opp_roshans"]
        dfm["totalbarracks"] = dfm["barracks"] + dfm["opp_barracks"]

        dfm["time_min"] = dfm["duration"] / 60.0 if "duration" in dfm.columns else np.nan

        dfm = dfm.sort_values("date", ascending=False).drop_duplicates(subset=["match_id"])
        return dfm


    # --------------- Match-level pair (H2H) and diffs (T1 - T2) ---------------

    def match_pair_h2h(self, df_scope: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
        """
        Retorna 1 linha por match_id (somente jogos H2H), com colunas:
          kills_t1, kills_t2, towers_t1, towers_t2, roshans_t1, roshans_t2, barracks_t1, barracks_t2,
          totalkills, totaltowers, ...
          diff_kills = t1 - t2 etc.
          time_min
        """
        t1n, t2n = norm_team(team1), norm_team(team2)

        mask = (
            df_scope["teamname_norm"].isin([t1n, t2n]) &
            df_scope["opp_teamname_norm"].isin([t1n, t2n])
        )
        df_h2h = df_scope[mask].copy()
        if df_h2h.empty:
            return df_h2h

        # Só queremos match_id onde existam as duas linhas (t1 e t2)
        # Pivot simples pela teamname_norm
        wanted_cols = [
            "match_id", "date", "league", "teamname_norm",
            "teamkills", "towers", "roshans", "barracks",
            "oppkills", "opp_towers", "opp_roshans", "opp_barracks",
            "duration"
        ]
        df_h2h = df_h2h[wanted_cols].copy()
        df_h2h = df_h2h.sort_values("date", ascending=False)

        # Se match_id não for numérico, pivot ainda funciona como chave
        pivot = df_h2h.pivot_table(
            index=["match_id", "date", "league"],
            columns="teamname_norm",
            values=["teamkills", "towers", "roshans", "barracks", "duration"],
            aggfunc="first"
        )

        # Garantir que existam colunas t1 e t2
        if t1n not in pivot["teamkills"].columns or t2n not in pivot["teamkills"].columns:
            return pd.DataFrame()

        out = pd.DataFrame(index=pivot.index).reset_index()
        out["kills_t1"] = pivot["teamkills"][t1n].values
        out["kills_t2"] = pivot["teamkills"][t2n].values
        out["towers_t1"] = pivot["towers"][t1n].values
        out["towers_t2"] = pivot["towers"][t2n].values
        out["roshans_t1"] = pivot["roshans"][t1n].values
        out["roshans_t2"] = pivot["roshans"][t2n].values
        out["barracks_t1"] = pivot["barracks"][t1n].values
        out["barracks_t2"] = pivot["barracks"][t2n].values

        # duration: pega do t1 se existir, senão do t2
        dur_t1 = pivot["duration"][t1n].values
        dur_t2 = pivot["duration"][t2n].values
        out["duration"] = np.where(~np.isnan(dur_t1), dur_t1, dur_t2)
        out["time_min"] = out["duration"] / 60.0

        out["totalkills"] = out["kills_t1"] + out["kills_t2"]
        out["totaltowers"] = out["towers_t1"] + out["towers_t2"]
        out["totalroshans"] = out["roshans_t1"] + out["roshans_t2"]
        out["totalbarracks"] = out["barracks_t1"] + out["barracks_t2"]

        out["diff_kills"] = out["kills_t1"] - out["kills_t2"]
        out["diff_towers"] = out["towers_t1"] - out["towers_t2"]
        out["diff_roshans"] = out["roshans_t1"] - out["roshans_t2"]
        out["diff_barracks"] = out["barracks_t1"] - out["barracks_t2"]

        out = out.sort_values("date", ascending=False)
        return out


# ==========================
# Analyzer (Over/Under + Handicap)
# ==========================

class DotaAnalyzer:
    def __init__(self, dataset: DotaDataset):
        self.ds = dataset

    def _slice_periods(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Retorna dict per período: Ano Todo, 10 jogos, 5 jogos
        df já deve estar ordenado por date desc.
        """
        if df is None or df.empty:
            return {"Ano Todo": df, "10 jogos": df, "5 jogos": df}
        df2 = df.sort_values("date", ascending=False)
        return {
            "Ano Todo": df2,
            "10 jogos": df2.head(10),
            "5 jogos": df2.head(5),
        }

    def _ou_counts(self, values: np.ndarray, line: float) -> Tuple[int, int, int]:
        """
        Retorna (over_wins, under_wins, pushes) com regra:
          over: val > line
          under: val < line
          push: val == line
        """
        if values.size == 0:
            return 0, 0, 0
        over = int(np.sum(values > line))
        under = int(np.sum(values < line))
        push = int(np.sum(values == line))
        return over, under, push


    def analyze(
        self,
        league: str,
        team1: str,
        team2: str,
        lines: LinesConfig,
        laplace: bool = True
    ) -> pd.DataFrame:
        """
        Retorna um DF pronto para a Treeview, com colunas:
          Análise | Ano Todo | 10 jogos | 5 jogos

        Padrão de amostragem/odd igual ao Gabinete LoL:
          "O: acertos/valid (odd.xx x)" / "U: ..." e, para HC, "C: ...".
        """
        if lines.is_empty():
            return pd.DataFrame([{"Análise": "Defina linhas em 'Adicionar/Mudar Linhas'."}])

        # Escopo da liga
        df_scope = self.ds.filter_league(league)

        # League sample: jogos do T1 + jogos do T2 (dedup por match_id)
        t1_tot = self.ds.match_totals_for_team(df_scope, team1)
        t2_tot = self.ds.match_totals_for_team(df_scope, team2)
        liga_tot = pd.concat([t1_tot, t2_tot], ignore_index=True)
        if not liga_tot.empty and "match_id" in liga_tot.columns:
            liga_tot = liga_tot.sort_values("date", ascending=False).drop_duplicates(subset=["match_id"])

        # H2H pair-level
        h2h = self.ds.match_pair_h2h(df_scope, team1, team2)

        # Particionar por períodos
        # Particionar por períodos
        # Liga: "10 jogos" e "5 jogos" seguem o padrão do Gabinete LoL:
        # últimos N do T1 + últimos N do T2 (não é "últimos N no total").
        p_liga = {
            "Ano Todo": liga_tot,
            "10 jogos": pd.concat([t1_tot.head(10), t2_tot.head(10)], ignore_index=True)
                       .sort_values("date", ascending=False)
                       .drop_duplicates(subset=["match_id"]) if (not t1_tot.empty or not t2_tot.empty) else pd.DataFrame(),
            "5 jogos": pd.concat([t1_tot.head(5), t2_tot.head(5)], ignore_index=True)
                      .sort_values("date", ascending=False)
                      .drop_duplicates(subset=["match_id"]) if (not t1_tot.empty or not t2_tot.empty) else pd.DataFrame(),
        }
        p_h2h = self._slice_periods(h2h)

        def safe_vals(df: pd.DataFrame, col: str) -> np.ndarray:
            if df is None or df.empty or col not in df.columns:
                return np.array([], dtype=float)
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            return v

        def fmt_cell(prefix: str, wins: int, total: int, pushes: int = 0) -> str:
            valid = max(0, int(total) - int(pushes))
            if valid <= 0:
                return "—"
            w = max(0, int(wins))
            if laplace:
                p = (w + 1) / (valid + 2)
            else:
                p = w / valid if valid else 0.0
            odd = float("inf") if p <= 0 else 1.0 / p
            return f"{prefix}: {w}/{valid} ({odd:.2f}x)"

        rows: List[dict] = []

        def add_ou_block(label: str, col_name: str, line_list: List[float], is_time: bool = False):
            if not line_list:
                return
            for line in line_list:
                line_str = fmt_mmss_from_min(line) if is_time else f"{line:g}"

                row_l_over = {"Análise": f"Liga - {label} Over ({line_str})"}
                row_l_under = {"Análise": f"Liga - {label} Under ({line_str})"}
                row_h_over = {"Análise": f"H2H - {label} Over ({line_str})"}
                row_h_under = {"Análise": f"H2H - {label} Under ({line_str})"}

                for period in ["Ano Todo", "10 jogos", "5 jogos"]:
                    # Liga (T1+T2)
                    dfp = p_liga.get(period)
                    vals = safe_vals(dfp, col_name)
                    over, under, push = self._ou_counts(vals, line)
                    total = len(vals)
                    row_l_over[period] = fmt_cell("O", over, total, push)
                    row_l_under[period] = fmt_cell("U", under, total, push)

                    # H2H
                    dfh = p_h2h.get(period)
                    vals_h = safe_vals(dfh, col_name)
                    overh, underh, pushh = self._ou_counts(vals_h, line)
                    totalh = len(vals_h)
                    row_h_over[period] = fmt_cell("O", overh, totalh, pushh)
                    row_h_under[period] = fmt_cell("U", underh, totalh, pushh)

                rows.extend([row_l_over, row_l_under, row_h_over, row_h_under])

        # Totais (O/U)
        add_ou_block("Kills", "totalkills", lines.total_kills)
        add_ou_block("Torres", "totaltowers", lines.total_towers)
        add_ou_block("Roshans", "totalroshans", lines.total_roshans)
        add_ou_block("Barracks", "totalbarracks", lines.total_barracks)
        add_ou_block("Tempo", "time_min", lines.total_time_min, is_time=True)

        # Handicaps (H2H somente, baseado em diff = T1 - T2)
        def add_hc_block(label: str, diff_col: str, line_list: List[float]):
            if not line_list:
                return
            for line in line_list:
                row_t1 = {"Análise": f"H2H - HC {label} T1 ({line:+g})"}
                row_t2 = {"Análise": f"H2H - HC {label} T2 ({(-line):+g})"}

                for period in ["Ano Todo", "10 jogos", "5 jogos"]:
                    dfh = p_h2h.get(period)
                    vals = safe_vals(dfh, diff_col)
                    total = len(vals)
                    thr = -line

                    wins_t1 = int(np.sum(vals > thr)) if total > 0 else 0
                    wins_t2 = int(np.sum(vals < thr)) if total > 0 else 0
                    pushes = int(np.sum(vals == thr)) if total > 0 else 0

                    row_t1[period] = fmt_cell("C", wins_t1, total, pushes)
                    row_t2[period] = fmt_cell("C", wins_t2, total, pushes)

                rows.extend([row_t1, row_t2])

        add_hc_block("Kills", "diff_kills", lines.hc_kills_t1)
        add_hc_block("Torres", "diff_towers", lines.hc_towers_t1)
        add_hc_block("Roshans", "diff_roshans", lines.hc_roshans_t1)
        add_hc_block("Barracks", "diff_barracks", lines.hc_barracks_t1)

        if not rows:
            return pd.DataFrame([{"Análise": "Sem linhas para calcular."}])

        df_out = pd.DataFrame(rows)

        # Garantir colunas
        for c in ["Análise", "Ano Todo", "10 jogos", "5 jogos"]:
            if c not in df_out.columns:
                df_out[c] = ""
        df_out = df_out[["Análise", "Ano Todo", "10 jogos", "5 jogos"]]
        return df_out


    def compute_overview(self, league: str, team1: str, team2: str) -> Dict[str, dict]:
        """
        Gera estatísticas base para a tela inicial (Visão Geral), por período:
          - H2H (confronto direto)
          - Comparativo: T1 vs T2 vs Média Geral (match-level no escopo)
        Retorna dict:
          {
            "Ano Todo": {...},
            "10 jogos": {...},
            "5 jogos": {...},
          }
        """
        df_scope = self.ds.filter_league(league)

        t1_tot = self.ds.match_totals_for_team(df_scope, team1)
        t2_tot = self.ds.match_totals_for_team(df_scope, team2)
        h2h = self.ds.match_pair_h2h(df_scope, team1, team2)

        p_t1 = self._slice_periods(t1_tot)
        p_t2 = self._slice_periods(t2_tot)
        p_h2h = self._slice_periods(h2h)

        def pack(df: pd.DataFrame) -> dict:
            if df is None or df.empty:
                return {
                    "games": 0,
                    "avg_kills": float("nan"),
                    "avg_towers": float("nan"),
                    "avg_roshans": float("nan"),
                    "avg_barracks": float("nan"),
                    "avg_time": float("nan"),
                }
            return {
                "games": int(len(df)),
                "avg_kills": float(np.nanmean(df.get("totalkills", np.nan))),
                "avg_towers": float(np.nanmean(df.get("totaltowers", np.nan))),
                "avg_roshans": float(np.nanmean(df.get("totalroshans", np.nan))),
                "avg_barracks": float(np.nanmean(df.get("totalbarracks", np.nan))),
                "avg_time": float(np.nanmean(df.get("time_min", np.nan))),
            }

        out: Dict[str, dict] = {}
        for period in ["Ano Todo", "10 jogos", "5 jogos"]:
            # H2H usa as colunas totais no próprio DF pair-level
            dfh = p_h2h[period]
            if dfh is None or dfh.empty:
                h2h_pack = {
                    "games": 0,
                    "avg_kills": float("nan"),
                    "avg_towers": float("nan"),
                    "avg_roshans": float("nan"),
                    "avg_barracks": float("nan"),
                    "avg_time": float("nan"),
                }
            else:
                h2h_pack = {
                    "games": int(len(dfh)),
                    "avg_kills": float(np.nanmean(dfh.get("totalkills", np.nan))),
                    "avg_towers": float(np.nanmean(dfh.get("totaltowers", np.nan))),
                    "avg_roshans": float(np.nanmean(dfh.get("totalroshans", np.nan))),
                    "avg_barracks": float(np.nanmean(dfh.get("totalbarracks", np.nan))),
                    "avg_time": float(np.nanmean(dfh.get("time_min", np.nan))),
                }

            # "Média Geral": junta os jogos do Time 1 + Time 2 no mesmo período
            d1 = p_t1[period]
            d2 = p_t2[period]
            if (d1 is None or d1.empty) and (d2 is None or d2.empty):
                df_all = pd.DataFrame()
            elif d1 is None or d1.empty:
                df_all = d2.copy()
            elif d2 is None or d2.empty:
                df_all = d1.copy()
            else:
                df_all = pd.concat([d1, d2], ignore_index=True)

            out[period] = {
                "h2h": h2h_pack,
                "t1": pack(p_t1[period]),
                "t2": pack(p_t2[period]),
                "all": pack(df_all),
            }
        return out



# ==========================
# UI - Lines Window
# ==========================

class LinesWindow(_ToplevelBase):
    def __init__(self, master: tk.Tk, current: LinesConfig, on_apply):
        if not _TK_AVAILABLE:
            raise RuntimeError(
                "Tkinter não está disponível neste ambiente. "
                "Para rodar no navegador, use o Streamlit (arquivo streamlit_app.py)."
            )
        super().__init__(master)
        self.title("Editar Todas as Linhas de Análise (Dota)")
        self.resizable(False, False)
        self.on_apply = on_apply

        self.vars: Dict[str, tk.StringVar] = {
            "total_kills": tk.StringVar(value=", ".join(map(str, current.total_kills)) if current.total_kills else ""),
            "total_towers": tk.StringVar(value=", ".join(map(str, current.total_towers)) if current.total_towers else ""),
            "total_roshans": tk.StringVar(value=", ".join(map(str, current.total_roshans)) if current.total_roshans else ""),
            "total_barracks": tk.StringVar(value=", ".join(map(str, current.total_barracks)) if current.total_barracks else ""),
            "total_time": tk.StringVar(
                value=", ".join([f"{v:.2f}" for v in current.total_time_min]) if current.total_time_min else ""
            ),
            "hc_kills": tk.StringVar(value=", ".join(map(str, current.hc_kills_t1)) if current.hc_kills_t1 else ""),
            "hc_towers": tk.StringVar(value=", ".join(map(str, current.hc_towers_t1)) if current.hc_towers_t1 else ""),
            "hc_roshans": tk.StringVar(value=", ".join(map(str, current.hc_roshans_t1)) if current.hc_roshans_t1 else ""),
            "hc_barracks": tk.StringVar(value=", ".join(map(str, current.hc_barracks_t1)) if current.hc_barracks_t1 else ""),
        }

        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        # Linhas de Totais
        frame_tot = ttk.LabelFrame(self, text="Linhas de Totais")
        frame_tot.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(frame_tot, text="Kills (ex: 45.5, 49.5)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame_tot, textvariable=self.vars["total_kills"], width=32).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frame_tot, text="Torres (ex: 18.5)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame_tot, textvariable=self.vars["total_towers"], width=32).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frame_tot, text="Roshans (ex: 2.5)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frame_tot, textvariable=self.vars["total_roshans"], width=32).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frame_tot, text="Barracks (ex: 4.5)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frame_tot, textvariable=self.vars["total_barracks"], width=32).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(frame_tot, text="Tempo (MM:SS ou min) (ex: 38:30, 42.5)").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frame_tot, textvariable=self.vars["total_time"], width=32).grid(row=4, column=1, sticky="w", **pad)

        # Linhas de Handicap
        frame_hc = ttk.LabelFrame(self, text="Linhas de Handicap")
        frame_hc.grid(row=1, column=0, sticky="ew", **pad)

        ttk.Label(frame_hc, text="HC Kills para T1 (ex: -2.5, -1.5)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame_hc, textvariable=self.vars["hc_kills"], width=32).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frame_hc, text="HC Torres para T1 (ex: -3.5)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame_hc, textvariable=self.vars["hc_towers"], width=32).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frame_hc, text="HC Roshans para T1 (ex: -0.5)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frame_hc, textvariable=self.vars["hc_roshans"], width=32).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frame_hc, text="HC Barracks para T1 (ex: -1.5)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frame_hc, textvariable=self.vars["hc_barracks"], width=32).grid(row=3, column=1, sticky="w", **pad)

        # Botão
        btn = ttk.Button(self, text="Confirmar e Atualizar Análise", command=self._apply)
        btn.grid(row=2, column=0, pady=(10, 10))

        # Tecla Enter confirma
        self.bind("<Return>", lambda e: self._apply())

    def _apply(self):
        cfg = LinesConfig(
            total_kills=parse_float_list(self.vars["total_kills"].get()),
            total_towers=parse_float_list(self.vars["total_towers"].get()),
            total_roshans=parse_float_list(self.vars["total_roshans"].get()),
            total_barracks=parse_float_list(self.vars["total_barracks"].get()),
            total_time_min=parse_time_line(self.vars["total_time"].get()),
            hc_kills_t1=parse_float_list(self.vars["hc_kills"].get()),
            hc_towers_t1=parse_float_list(self.vars["hc_towers"].get()),
            hc_roshans_t1=parse_float_list(self.vars["hc_roshans"].get()),
            hc_barracks_t1=parse_float_list(self.vars["hc_barracks"].get()),
        )
        self.on_apply(cfg)
        self.destroy()


# ==========================
# UI - Main App
# ==========================

class GabineteDotaApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analisador de Estatísticas e Previsões - Dota")
        self.root.geometry("1280x720")

        self.lines_cfg = LinesConfig()
        self.laplace_var = tk.BooleanVar(value=True)

        self._combo_full_values = {}

        try:
            self.ds = DotaDataset(CSV_NAME)
            self.analyzer = DotaAnalyzer(self.ds)
        except Exception as e:
            messagebox.showerror("Erro ao carregar CSV", str(e))
            self.ds = None
            self.analyzer = None

        # Salvar jogos (confrontos)
        self.saved_games = {}
        self._load_saved_games()

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        # Tabs (mantém vibe do Gabinete)
        self.tab_confronto = ttk.Frame(self.notebook)
        self.tab_prev_series = ttk.Frame(self.notebook)
        self.tab_series_hist = ttk.Frame(self.notebook)
        self.tab_saved = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)
        self.tab_camp_stats = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_confronto, text="Análise de Confronto")
        self.notebook.add(self.tab_prev_series, text="Previsão de Séries")
        self.notebook.add(self.tab_series_hist, text="Análise de Séries (Histórico)")
        self.notebook.add(self.tab_log, text="Log de Carga")
        self.notebook.add(self.tab_camp_stats, text="Camp Betting Stats")
        self.notebook.add(self.tab_saved, text="Salvar Jogos")

        self._build_tab_confronto()
        self._build_tab_prev_series()
        self._build_placeholder_tab(self.tab_series_hist, "Análise de Séries (Histórico) (em desenvolvimento)")
        self._build_tab_log()
        self._build_tab_camp_stats()
        self._build_tab_saved()


    def _build_tab_saved(self):
        frm = ttk.Frame(self.tab_saved, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Confrontos Salvos", padding=10)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Aqui você salva o confronto (liga + times) e depois pode completar com as linhas.").grid(
            row=0, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 8)
        )

        ttk.Button(top, text="Salvar Confronto Atual", command=self._btn_save_current).grid(row=1, column=0, padx=6, pady=6)
        ttk.Button(top, text="Carregar para Análise", command=self._btn_load_saved).grid(row=1, column=1, padx=6, pady=6)
        ttk.Button(top, text="Editar Linhas do Jogo", command=self._btn_edit_saved_lines).grid(row=1, column=2, padx=6, pady=6)
        ttk.Button(top, text="Excluir", command=self._btn_delete_saved).grid(row=1, column=3, padx=6, pady=6)
        ttk.Button(top, text="Atualizar Lista", command=self._refresh_saved_tree).grid(row=1, column=4, padx=6, pady=6)

        # Tabela
        table_frame = ttk.LabelFrame(frm, text="Lista", padding=10)
        table_frame.pack(side="top", fill="both", expand=True, pady=(10, 0))

        cols = ("updated_at", "league", "team1", "team2", "has_lines", "key")
        self.saved_tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        self.saved_tree.heading("updated_at", text="Atualizado")
        self.saved_tree.heading("league", text="Liga")
        self.saved_tree.heading("team1", text="Time 1")
        self.saved_tree.heading("team2", text="Time 2")
        self.saved_tree.heading("has_lines", text="Linhas?")
        self.saved_tree.heading("key", text="Key (interno)")

        self.saved_tree.column("updated_at", width=160, anchor="w")
        self.saved_tree.column("league", width=260, anchor="w")
        self.saved_tree.column("team1", width=240, anchor="w")
        self.saved_tree.column("team2", width=240, anchor="w")
        self.saved_tree.column("has_lines", width=70, anchor="center")
        self.saved_tree.column("key", width=1, stretch=False)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.saved_tree.yview)
        self.saved_tree.configure(yscrollcommand=vsb.set)

        self.saved_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # duplo clique -> carregar
        self.saved_tree.bind("<Double-Button-1>", lambda e: self._btn_load_saved())

        self._refresh_saved_tree()

    def _btn_save_current(self):
        league, t1, t2 = self._get_selection()
        if not t1 or not t2:
            messagebox.showwarning("Atenção", "Selecione Time 1 e Time 2 para salvar.")
            return
        if norm_team(t1) == norm_team(t2):
            messagebox.showwarning("Atenção", "Selecione dois times diferentes.")
            return

        # salva sem obrigar linhas
        self._upsert_saved_game(league, t1, t2, lines=self.lines_cfg if not self.lines_cfg.is_empty() else None)
        messagebox.showinfo("OK", "Confronto salvo!")

    def _btn_load_saved(self):
        key = self._get_selected_saved_key()
        if not key:
            messagebox.showwarning("Atenção", "Selecione um confronto salvo.")
            return
        rec = self.saved_games.get(key)
        if not rec:
            return
        # muda pra aba de análise
        self.notebook.select(self.tab_confronto)
        self._load_saved_into_selection(rec)

    def _btn_edit_saved_lines(self):
        key = self._get_selected_saved_key()
        if not key:
            messagebox.showwarning("Atenção", "Selecione um confronto salvo.")
            return
        rec = self.saved_games.get(key)
        if not rec:
            return
        self._edit_saved_lines(rec)

    def _btn_delete_saved(self):
        key = self._get_selected_saved_key()
        if not key:
            messagebox.showwarning("Atenção", "Selecione um confronto salvo.")
            return
        rec = self.saved_games.get(key)
        if not rec:
            return
        ok = messagebox.askyesno("Confirmar", f"Excluir o confronto salvo?\n\n{rec.get('team1')} x {rec.get('team2')}")
        if not ok:
            return
        self.saved_games.pop(key, None)
        self._persist_saved_games()
        self._refresh_saved_tree()


    def _build_placeholder_tab(self, tab, text: str):
        frm = ttk.Frame(tab, padding=14)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=text).pack(anchor="w")


    def _build_tab_series_forecast(self):
        frm = ttk.Frame(self.tab_series_forecast, padding=10)
        frm.pack(fill="both", expand=True)
        box = ttk.LabelFrame(frm, text="Previsão de Séries (em construção)", padding=12)
        box.pack(fill="both", expand=True)
        ttk.Label(
            box,
            text="Esta aba foi criada para ficar igual ao GabineteFINAL3.0.\n\nPróximo passo: adaptar previsão de séries para Dota (BO1/BO3/BO5) e odds/EV.",
            justify="left"
        ).pack(anchor="w")


    def _build_tab_series_history(self):
        frm = ttk.Frame(self.tab_series_history, padding=10)
        frm.pack(fill="both", expand=True)
        box = ttk.LabelFrame(frm, text="Análise de Séries (Histórico) (em construção)", padding=12)
        box.pack(fill="both", expand=True)
        ttk.Label(
            box,
            text="Esta aba foi criada para ficar igual ao GabineteFINAL3.0.\n\nPróximo passo: histórico de séries por time/liga e métricas agregadas.",
            justify="left"
        ).pack(anchor="w")


    def _build_tab_camp_stats(self):
        frm = ttk.Frame(self.tab_camp_stats, padding=10)
        frm.pack(fill="both", expand=True)
        box = ttk.LabelFrame(frm, text="Camp Betting Stats (em construção)", padding=12)
        box.pack(fill="both", expand=True)
        ttk.Label(
            box,
            text="Esta aba foi criada para ficar igual ao GabineteFINAL3.0.\n\nPróximo passo: estatísticas por campeonato (liga) + consistência de mercado.",
            justify="left"
        ).pack(anchor="w")


    def _build_tab_log(self):
        frm = ttk.Frame(self.tab_log, padding=10)
        frm.pack(fill="both", expand=True)
        self.txt_log = ScrolledText(frm, height=10)
        self.txt_log.pack(fill="both", expand=True)
        self._log(f"CSV esperado: {CSV_NAME} (na mesma pasta do .py)\n")


    def _build_tab_camp_stats(self):
        """
        Camp Betting Stats (estilo GabineteFINAL3.0):
        - Filtros por campeonato (league) e ano
        - Busca rápida por Time e por Liga
        - Tabela agregada por (Liga, Time) com métricas e % de linhas típicas
        """
        frm = ttk.Frame(self.tab_camp_stats, padding=10)
        frm.pack(fill="both", expand=True)

        container = ttk.LabelFrame(frm, text="Camp Betting Stats", padding=10)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Base
        base = getattr(self.ds, "df", None)
        if base is None or len(base) == 0:
            ttk.Label(container, text="Sem dados carregados do CSV.").pack(anchor="w", pady=10)
            return

        df = base.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Checagens mínimas
        required = {"league", "year", "teamname", "result", "duration",
                    "teamkills", "oppkills",
                    "towers", "opp_towers",
                    "roshans", "opp_roshans",
                    "barracks", "opp_barracks"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            ttk.Label(
                container,
                text="CSV sem colunas necessárias para Camp Stats:\n" + ", ".join(missing),
                justify="left"
            ).pack(anchor="w", pady=10)
            return

        # Listas de filtros
        leagues = sorted([str(x) for x in pd.Series(df["league"].dropna().unique()).tolist() if str(x).strip() != ""])
        years = sorted(pd.Series(df["year"].dropna().unique()).astype(int).tolist())

        # Vars (persistem na sessão)
        if not hasattr(self, "var_camp_team_search"):
            self.var_camp_team_search = tk.StringVar()
        if not hasattr(self, "var_camp_league_search"):
            self.var_camp_league_search = tk.StringVar()
        if not hasattr(self, "var_camp_league"):
            self.var_camp_league = tk.StringVar(value="Todos os campeonatos")
        if not hasattr(self, "var_camp_year"):
            self.var_camp_year = tk.StringVar(value="Todos os anos")

        # ---- Barra de filtros ----
        filters = ttk.Frame(container)
        filters.pack(fill="x", pady=(0, 8))

        ttk.Label(filters, text="Time:").pack(side="left", padx=(8, 2))
        entry_team = ttk.Entry(filters, textvariable=self.var_camp_team_search, width=22)
        entry_team.pack(side="left", padx=(0, 4))

        ttk.Button(filters, text="🔎", width=3, command=lambda: _refresh()).pack(side="left")

        ttk.Label(filters, text="Liga (busca):").pack(side="left", padx=(10, 2))
        entry_league = ttk.Entry(filters, textvariable=self.var_camp_league_search, width=18)
        entry_league.pack(side="left", padx=(0, 4))
        ttk.Button(filters, text="🔎", width=3, command=lambda: _refresh()).pack(side="left")

        ttk.Label(filters, text="Campeonato:").pack(side="left", padx=(12, 6))
        cb_league = ttk.Combobox(
            filters,
            textvariable=self.var_camp_league,
            state="readonly",
            values=["Todos os campeonatos"] + leagues,
            width=40
        )
        cb_league.pack(side="left")

        ttk.Label(filters, text="Ano:").pack(side="left", padx=(12, 6))
        cb_year = ttk.Combobox(
            filters,
            textvariable=self.var_camp_year,
            state="readonly",
            values=["Todos os anos"] + [str(y) for y in years],
            width=14
        )
        cb_year.pack(side="left")

        def _next_prev(delta: int):
            vals = ["Todos os campeonatos"] + leagues
            try:
                idx = vals.index(self.var_camp_league.get())
            except ValueError:
                idx = 0
            idx = (idx + delta) % len(vals)
            self.var_camp_league.set(vals[idx])
            _refresh()

        ttk.Button(filters, text="◀", width=3, command=lambda: _next_prev(-1)).pack(side="left", padx=(12, 4))
        ttk.Button(filters, text="▶", width=3, command=lambda: _next_prev(+1)).pack(side="left", padx=(0, 8))

        ttk.Button(filters, text="Atualizar", command=lambda: _refresh()).pack(side="left", padx=(0, 8))

        # Tecla Enter atualiza
        entry_team.bind("<Return>", lambda e: _refresh())
        entry_league.bind("<Return>", lambda e: _refresh())
        cb_league.bind("<<ComboboxSelected>>", lambda e: _refresh())
        cb_year.bind("<<ComboboxSelected>>", lambda e: _refresh())

        # ---- Tabela ----
        table_frame = ttk.Frame(container)
        table_frame.pack(fill="both", expand=True)

        cols = [
            "League", "Team", "# Games", "Win Rate", "Game Time",
            "Kills", "Deaths", "Total Kills",
            "Towers", "Total Towers",
            "Roshans", "Total Roshans",
            "Barracks", "Total Barracks",
            "C. Kills", "C. Towers", "C. Roshans", "C. Barracks",
            "% TK > 45.5", "% TK > 50.5", "% TK > 55.5",
            "% TT > 15.5", "% TR > 1.5", "% TB > 3.5",
            "% Time > 40.5", "% Time > 45.5"
        ]

        tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # Configura cabeçalhos e colunas
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="center", stretch=True)
        tree.column("League", width=220, anchor="w")
        tree.column("Team", width=220, anchor="w")
        tree.column("# Games", width=80)
        tree.column("Win Rate", width=90)
        tree.column("Game Time", width=90)

        def _fmt_time_seconds(sec):
            try:
                if sec is None or (isinstance(sec, float) and math.isnan(sec)):
                    return ""
                sec = float(sec)
            except Exception:
                return ""
            m = int(sec // 60)
            s = int(round(sec - m * 60))
            if s >= 60:
                m += 1
                s -= 60
            return f"{m:02d}:{s:02d}"

        def _fmt_value(col, val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return ""
            if col == "Game Time":
                return _fmt_time_seconds(val)
            if col in ("Win Rate", "% TK > 45.5", "% TK > 50.5", "% TK > 55.5",
                       "% TT > 15.5", "% TR > 1.5", "% TB > 3.5",
                       "% Time > 40.5", "% Time > 45.5"):
                try:
                    return f"{float(val):.1f}%"
                except Exception:
                    return str(val)
            if col == "# Games":
                try:
                    return str(int(val))
                except Exception:
                    return str(val)
            # numérico geral
            try:
                f = float(val)
                # se for "quase inteiro", mostra sem decimal
                if abs(f - round(f)) < 1e-9 and abs(f) < 1e6:
                    return str(int(round(f)))
                return f"{f:.2f}"
            except Exception:
                return str(val)

        current_df = None
        sort_state = {"col": None, "reverse": False}

        def _fill_tree(df_fill: pd.DataFrame):
            tree.delete(*tree.get_children())
            if df_fill is None or df_fill.empty:
                return
            for _, row in df_fill.iterrows():
                tree.insert("", "end", values=[_fmt_value(c, row.get(c, "")) for c in cols])

        def _safe_pct(series_bool: pd.Series):
            try:
                s = series_bool.dropna()
                if len(s) == 0:
                    return np.nan
                return float(s.mean() * 100.0)
            except Exception:
                return np.nan

        def _make_stats(df_in: pd.DataFrame) -> pd.DataFrame:
            # linhas típicas (ajuste fácil depois)
            tk_lines = [45.5, 50.5, 55.5]   # Total Kills
            tt_line = 15.5                 # Total Towers
            tr_line = 1.5                  # Total Roshans
            tb_line = 3.5                  # Total Barracks
            tmin_lines = [40.5, 45.5]      # Tempo (min)

            rows = []
            # agrega por Liga + Time (importante quando "Todos os campeonatos")
            grp = df_in.groupby(["league", "teamname_norm"] if "teamname_norm" in df_in.columns else ["league", "teamname"], dropna=False)
            for (lg, _tn), g in grp:
                if g is None or g.empty:
                    continue

                # display name do time
                try:
                    team_disp = g["teamname"].mode().iloc[0]
                except Exception:
                    team_disp = str(g["teamname"].iloc[0]) if "teamname" in g.columns else str(_tn)

                n = int(len(g))
                winr = float(pd.to_numeric(g["result"], errors="coerce").fillna(0).mean() * 100.0)

                dur = pd.to_numeric(g["duration"], errors="coerce")
                dur_sec = float(dur.mean()) if dur.notna().any() else np.nan

                kills = pd.to_numeric(g["teamkills"], errors="coerce")
                deaths = pd.to_numeric(g["oppkills"], errors="coerce")
                towers = pd.to_numeric(g["towers"], errors="coerce")
                rosh = pd.to_numeric(g["roshans"], errors="coerce")
                barr = pd.to_numeric(g["barracks"], errors="coerce")

                ok = lambda s: float(s.mean()) if s.notna().any() else np.nan

                tot_k = kills + deaths
                tot_t = towers + pd.to_numeric(g["opp_towers"], errors="coerce")
                tot_r = rosh + pd.to_numeric(g["opp_roshans"], errors="coerce")
                tot_b = barr + pd.to_numeric(g["opp_barracks"], errors="coerce")

                # diffs
                c_k = kills - deaths
                c_t = towers - pd.to_numeric(g["opp_towers"], errors="coerce")
                c_r = rosh - pd.to_numeric(g["opp_roshans"], errors="coerce")
                c_b = barr - pd.to_numeric(g["opp_barracks"], errors="coerce")

                dur_min = dur / 60.0

                row = {
                    "League": str(lg),
                    "Team": str(team_disp),
                    "# Games": n,
                    "Win Rate": winr,
                    "Game Time": dur_sec,

                    "Kills": ok(kills),
                    "Deaths": ok(deaths),
                    "Total Kills": ok(tot_k),

                    "Towers": ok(towers),
                    "Total Towers": ok(tot_t),

                    "Roshans": ok(rosh),
                    "Total Roshans": ok(tot_r),

                    "Barracks": ok(barr),
                    "Total Barracks": ok(tot_b),

                    "C. Kills": ok(c_k),
                    "C. Towers": ok(c_t),
                    "C. Roshans": ok(c_r),
                    "C. Barracks": ok(c_b),
                }

                # percentuais de linhas
                row["% TK > 45.5"] = _safe_pct(tot_k > tk_lines[0])
                row["% TK > 50.5"] = _safe_pct(tot_k > tk_lines[1])
                row["% TK > 55.5"] = _safe_pct(tot_k > tk_lines[2])
                row["% TT > 15.5"] = _safe_pct(tot_t > tt_line)
                row["% TR > 1.5"] = _safe_pct(tot_r > tr_line)
                row["% TB > 3.5"] = _safe_pct(tot_b > tb_line)

                if dur_min.notna().any():
                    row["% Time > 40.5"] = _safe_pct(dur_min > tmin_lines[0])
                    row["% Time > 45.5"] = _safe_pct(dur_min > tmin_lines[1])
                else:
                    row["% Time > 40.5"] = np.nan
                    row["% Time > 45.5"] = np.nan

                rows.append(row)

            out = pd.DataFrame(rows)
            if out.empty:
                return out

            # ordenação padrão: por liga e win rate desc
            try:
                out = out.sort_values(["League", "Win Rate", "# Games"], ascending=[True, False, False]).reset_index(drop=True)
            except Exception:
                pass
            return out

        def _refresh():
            nonlocal current_df
            try:
                sel_league = self.var_camp_league.get()
            except Exception:
                sel_league = "Todos os campeonatos"

            try:
                sel_year = self.var_camp_year.get()
            except Exception:
                sel_year = "Todos os anos"

            df_scope = df
            if sel_league and sel_league != "Todos os campeonatos":
                df_scope = df_scope[df_scope["league"] == sel_league]
            if sel_year and sel_year != "Todos os anos":
                try:
                    y = int(sel_year)
                    df_scope = df_scope[df_scope["year"].astype(int) == y]
                except Exception:
                    pass

            # calcula tabela
            current_df = _make_stats(df_scope)

            # filtros de busca
            ql = (self.var_camp_league_search.get() or "").strip()
            if ql and current_df is not None and not current_df.empty:
                current_df = current_df[current_df["League"].astype(str).str.contains(ql, case=False, na=False)].reset_index(drop=True)

            q = (self.var_camp_team_search.get() or "").strip()
            if q and current_df is not None and not current_df.empty:
                current_df = current_df[current_df["Team"].astype(str).str.contains(q, case=False, na=False)].reset_index(drop=True)

            _fill_tree(current_df)

        def _sort(col: str):
            nonlocal current_df, sort_state
            if current_df is None or current_df.empty:
                return
            reverse = (not sort_state["reverse"]) if sort_state["col"] == col else False
            sort_state = {"col": col, "reverse": reverse}

            try:
                if col in ("League", "Team"):
                    ordered = current_df.sort_values(col, ascending=not reverse)
                else:
                    ordered = current_df.sort_values(col, ascending=not reverse, na_position="last")
            except Exception:
                ordered = current_df
            _fill_tree(ordered)

        # clique no header ordena
        for c in cols:
            tree.heading(c, text=c, command=lambda col=c: _sort(col))

        _refresh()


    def _log(self, msg: str):
        if hasattr(self, "txt_log"):
            self.txt_log.insert("end", msg)
            if not msg.endswith("\n"):
                self.txt_log.insert("end", "\n")
            self.txt_log.see("end")

    def _norm_simple(self, s: str) -> str:
        s0 = _strip_accents(str(s or "")).lower()
        s0 = re.sub(r"\s+", " ", s0).strip()
        return s0

    def _set_combo_values(self, cb: ttk.Combobox, values: List[str]):
        # guarda a lista cheia pra busca/filtragem
        self._combo_full_values[cb] = list(values or [])
        cb["values"] = self._combo_full_values[cb]

    def _make_combo_searchable(self, cb: ttk.Combobox, max_show: int = 300):
        """
        Permite digitar no combobox e filtrar a lista enquanto digita (QOL).
        - Não perde a lista completa (fica em self._combo_full_values[cb]).
        - Limita o dropdown a max_show itens para não travar com listas gigantes.
        """
        def on_keyrelease(event):
            if event.keysym in ("Up", "Down", "Left", "Right", "Escape", "Return", "Tab"):
                return

            full = self._combo_full_values.get(cb)
            if full is None:
                full = list(cb.cget("values") or [])
                self._combo_full_values[cb] = full

            typed = cb.get()
            needle = self._norm_simple(typed)

            if not needle:
                cb["values"] = full
                return

            filtered = [v for v in full if needle in self._norm_simple(v)]
            shown = filtered[:max_show] if filtered else full[:max_show]
            cb["values"] = shown

            # tentativa de abrir o dropdown automaticamente (depende do Tk)
            # Só abre quando o usuário digitou algo "útil" e há matches.
            # (auto-abrir dropdown desativado para não atrapalhar a digitação)

        cb.bind("<KeyRelease>", on_keyrelease, add=True)

    def _resolve_from_list(self, typed: str, full_list: List[str]) -> str:
        """
        Resolve texto digitado para um item real da lista:
        - match exato (normalizado) -> usa item
        - contém (normalizado) com resultado único -> usa item
        - caso contrário, devolve o próprio texto digitado.
        Sempre retorna string.
        """
        typed = (typed or "").strip()
        if not typed:
            return ""

        # se já está exatamente na lista, ok
        if typed in full_list:
            return typed

        needle = self._norm_simple(typed)

        # match exato (normalizado)
        exact = [v for v in full_list if self._norm_simple(v) == needle]
        if len(exact) == 1:
            return exact[0]

        # contém (normalizado) com resultado único
        contains = [v for v in full_list if needle in self._norm_simple(v)]
        if len(contains) == 1:
            return contains[0]

        return typed


    def _choose_from_list(self, title: str, typed: str, options: List[str]) -> Optional[str]:
        """
        Abre uma janelinha com lista de opções (matches) para o usuário escolher.
        Retorna o item escolhido ou None (cancelado).
        """
        if not options:
            return None

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("520x420")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=f"Não encontrei um match único para: '{typed}'. Selecione abaixo:").pack(anchor="w", padx=10, pady=(10, 6))

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        lb = tk.Listbox(frame, height=15)
        sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)

        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for opt in options:
            lb.insert("end", opt)

        # pré-seleciona o primeiro
        if options:
            lb.selection_set(0)
            lb.activate(0)

        chosen = {"value": None}

        def confirm():
            sel = lb.curselection()
            if not sel:
                return
            chosen["value"] = lb.get(sel[0])
            win.destroy()

        def cancel():
            chosen["value"] = None
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(btns, text="OK", command=confirm).pack(side="left")
        ttk.Button(btns, text="Cancelar", command=cancel).pack(side="left", padx=(10, 0))

        lb.bind("<Double-Button-1>", lambda e: confirm())
        win.bind("<Return>", lambda e: confirm())
        win.bind("<Escape>", lambda e: cancel())

        self.root.wait_window(win)
        return chosen["value"]
        if typed in full_list:
            return typed

        needle = self._norm_simple(typed)
        exact = [v for v in full_list if self._norm_simple(v) == needle]
        if len(exact) == 1:
            return exact[0]

        contains = [v for v in full_list if needle in self._norm_simple(v)]
        if len(contains) == 1:
            return contains[0]

        return typed


    # ==========================
    # Salvar Jogos (Confrontos)
    # ==========================

    def _saved_file_path(self) -> str:
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, SAVED_GAMES_FILE)

    def _make_game_key(self, league: str, t1: str, t2: str) -> str:
        # chave independente da ordem (Heroic x Pari e Pari x Heroic vira o mesmo)
        ln = norm_team(league or "")
        a = norm_team(t1 or "")
        b = norm_team(t2 or "")
        pair = "::".join(sorted([a, b]))
        return f"{ln}::{pair}"

    def _load_saved_games(self):
        self.saved_games = {}
        try:
            fp = self._saved_file_path()
            if not os.path.exists(fp):
                return
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f) or []
            if isinstance(data, dict):
                # compat
                data = list(data.values())
            for item in data:
                key = item.get("key")
                if key:
                    self.saved_games[key] = item
        except Exception as e:
            self._log(f"[Salvar Jogos] Falha ao carregar: {e}\n")

    def _persist_saved_games(self):
        try:
            fp = self._saved_file_path()
            data = list(self.saved_games.values())
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"[Salvar Jogos] Falha ao salvar: {e}\n")

    def _record_has_lines(self, rec: dict) -> bool:
        d = rec.get("lines") or {}
        cfg = LinesConfig.from_dict(d)
        return not cfg.is_empty()

    def _upsert_saved_game(self, league: str, t1: str, t2: str, lines: Optional[LinesConfig] = None):
        key = self._make_game_key(league, t1, t2)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prev = self.saved_games.get(key) or {}
        prev_lines = prev.get("lines") or {}

        rec = {
            "key": key,
            "league": league,
            "team1": t1,
            "team2": t2,
            "created_at": prev.get("created_at") or now,
            "updated_at": now,
            "lines": (lines.to_dict() if lines else prev_lines),
        }
        self.saved_games[key] = rec
        self._persist_saved_games()
        self._refresh_saved_tree()

    def _get_selected_saved_key(self) -> Optional[str]:
        sel = self.saved_tree.selection()
        if not sel:
            return None
        item_id = sel[0]
        key = self.saved_tree.set(item_id, "key")
        return key or None

    def _refresh_saved_tree(self):
        if not hasattr(self, "saved_tree"):
            return
        for item in self.saved_tree.get_children():
            self.saved_tree.delete(item)

        # ordenar por updated_at desc
        def sort_key(rec):
            return rec.get("updated_at") or ""

        records = sorted(self.saved_games.values(), key=sort_key, reverse=True)
        for rec in records:
            has_lines = "✓" if self._record_has_lines(rec) else ""
            vals = (
                rec.get("updated_at", ""),
                rec.get("league", ""),
                rec.get("team1", ""),
                rec.get("team2", ""),
                has_lines,
                rec.get("key", ""),
            )
            self.saved_tree.insert("", "end", values=vals)

    def _load_saved_into_selection(self, rec: dict):
        league = (rec.get("league") or "(Todas)").strip() or "(Todas)"
        t1 = (rec.get("team1") or "").strip()
        t2 = (rec.get("team2") or "").strip()

        # set liga
        self.combo_liga.set(league if league in (self._combo_full_values.get(self.combo_liga, []) or []) else "(Todas)")
        self._update_teams()

        # set times
        self.combo_t1.set(t1)
        self.combo_t2.set(t2)

        # aplicar linhas salvas (se houver)
        cfg = LinesConfig.from_dict(rec.get("lines") or {})
        self.lines_cfg = cfg

        if not cfg.is_empty():
            self.on_analisar()
        else:
            self._log("Jogo carregado sem linhas. Abra 'Adicionar/Mudar Linhas' para completar.\n")

    def _edit_saved_lines(self, rec: dict):
        current_cfg = LinesConfig.from_dict(rec.get("lines") or {})

        def apply(cfg: LinesConfig):
            # salva
            rec["lines"] = cfg.to_dict()
            rec["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.saved_games[rec["key"]] = rec
            self._persist_saved_games()
            self._refresh_saved_tree()

            # se o jogo carregado na tela for o mesmo, aplica e recalcula
            league_now, t1_now, t2_now = self._get_selection()
            key_now = self._make_game_key(league_now, t1_now, t2_now) if t1_now and t2_now else None
            if key_now == rec["key"]:
                self.lines_cfg = cfg
                self.on_analisar()

        LinesWindow(self.root, current_cfg, apply)




    def _build_tab_confronto(self):
        # --- Top: seleção de times ---
        top = ttk.LabelFrame(self.tab_confronto, text="Seleção de Times", padding=10)
        top.pack(side="top", fill="x", padx=10, pady=10)

        ttk.Label(top, text="Liga:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.combo_liga = ttk.Combobox(top, state="normal", width=40)
        self.combo_liga.grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(top, text="Time 1:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.combo_t1 = ttk.Combobox(top, state="normal", width=40)
        self.combo_t1.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(top, text="Time 2:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.combo_t2 = ttk.Combobox(top, state="normal", width=40)
        self.combo_t2.grid(row=2, column=1, sticky="w", padx=6, pady=6)

        # Botões (estilo Gabinete)
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=0, column=2, rowspan=3, sticky="nw", padx=12, pady=6)

        ttk.Button(btn_frame, text="Analisar Confronto", command=self.on_analisar).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(btn_frame, text="Adicionar/Mudar Linhas", command=self.on_linhas).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(btn_frame, text="Gerar Resumo", command=self.on_resumo).grid(row=0, column=2, padx=6, pady=6)

        chk = ttk.Checkbutton(btn_frame, text="Laplace (suavizar odds)", variable=self.laplace_var)
        chk.grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        # Pesquisa: deixar digitar livremente. A resolução/sugestões acontecem só ao clicar em "Analisar Confronto".

        # popular listas
        self._populate_lists()

        self.combo_liga.bind("<<ComboboxSelected>>", lambda e: self._update_teams())
        self.combo_liga.bind("<Return>", lambda e: self._update_teams())
        self.combo_liga.bind("<FocusOut>", lambda e: self._update_teams())

        # --- Área principal com abas internas (igual vibe do LoL) ---
        main_area = ttk.Frame(self.tab_confronto, padding=8)
        main_area.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        self.nb_confronto = ttk.Notebook(main_area)
        self.nb_confronto.pack(fill="both", expand=True)

        self.tab_visao_geral = ttk.Frame(self.nb_confronto)
        self.tab_primeiros_obj = ttk.Frame(self.nb_confronto)
        self.tab_consistencia = ttk.Frame(self.nb_confronto)

        self.nb_confronto.add(self.tab_visao_geral, text="Visão Geral")
        self.nb_confronto.add(self.tab_primeiros_obj, text="Primeiros Objetivos")
        self.nb_confronto.add(self.tab_consistencia, text="Consistência de Mercado")

        # =========================
        # Visão Geral: Ano / Últimos 10 / Últimos 5
        # =========================
        self.nb_overview = ttk.Notebook(self.tab_visao_geral)
        self.nb_overview.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_over_year = ttk.Frame(self.nb_overview)
        self.tab_over_last10 = ttk.Frame(self.nb_overview)
        self.tab_over_last5 = ttk.Frame(self.nb_overview)

        self.nb_overview.add(self.tab_over_year, text="Ano Todo")
        self.nb_overview.add(self.tab_over_last10, text="Últimos 10")
        self.nb_overview.add(self.tab_over_last5, text="Últimos 5")

        def _build_overview_scope(parent, label_cmp="Comparativo de Estatísticas no Ano"):
            cont = ttk.Frame(parent, padding=10)
            cont.pack(fill="both", expand=True)

            h2h_frame = ttk.LabelFrame(cont, text="Estatísticas e Análises do Confronto Direto (H2H)", padding=10)
            h2h_frame.pack(side="top", fill="both", expand=True)

            cols_h2h = ("Métrica", "H2H")
            tree_h2h = ttk.Treeview(h2h_frame, columns=cols_h2h, show="headings", height=6)
            tree_h2h.heading("Métrica", text="Métrica")
            tree_h2h.heading("H2H", text="H2H")
            tree_h2h.column("Métrica", width=420, anchor="w")
            tree_h2h.column("H2H", width=140, anchor="center")

            vsb_h2h = ttk.Scrollbar(h2h_frame, orient="vertical", command=tree_h2h.yview)
            tree_h2h.configure(yscrollcommand=vsb_h2h.set)
            tree_h2h.pack(side="left", fill="both", expand=True)
            vsb_h2h.pack(side="right", fill="y")

            cmp_frame = ttk.LabelFrame(cont, text=label_cmp, padding=10)
            cmp_frame.pack(side="top", fill="both", expand=True, pady=(10, 0))

            cols_cmp = ("Métrica", "T1", "T2", "MÉDIA GERAL")
            tree_cmp = ttk.Treeview(cmp_frame, columns=cols_cmp, show="headings")
            for c in cols_cmp:
                tree_cmp.heading(c, text=c)

            tree_cmp.column("Métrica", width=420, anchor="w")
            tree_cmp.column("T1", width=160, anchor="center")
            tree_cmp.column("T2", width=160, anchor="center")
            tree_cmp.column("MÉDIA GERAL", width=160, anchor="center")

            vsb_cmp = ttk.Scrollbar(cmp_frame, orient="vertical", command=tree_cmp.yview)
            tree_cmp.configure(yscrollcommand=vsb_cmp.set)
            tree_cmp.pack(side="left", fill="both", expand=True)
            vsb_cmp.pack(side="right", fill="y")

            return tree_h2h, tree_cmp

        self._overview_trees = {
            "year": _build_overview_scope(self.tab_over_year, label_cmp="Comparativo de Estatísticas no Ano"),
            "last10": _build_overview_scope(self.tab_over_last10, label_cmp="Comparativo (Últimos 10)"),
            "last5": _build_overview_scope(self.tab_over_last5, label_cmp="Comparativo (Últimos 5)"),
        }

        # =========================
        # Primeiros Objetivos (placeholder por enquanto)
        # =========================
        fo = ttk.LabelFrame(self.tab_primeiros_obj, text="Primeiros Objetivos (em construção)", padding=12)
        fo.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(
            fo,
            text="Aqui vamos colocar futuramente: First Roshan, First Tower, etc.\nPor enquanto focamos no núcleo: kills/torres/roshans/barracks/tempo.",
            justify="left"
        ).pack(anchor="w")

        
        # =========================
        # Consistência de Mercado (O/U + HC) - agora separado por abas (Kills/Torres/Roshans/Barracks/Tempo/HC)
        # =========================
        market_frame = ttk.LabelFrame(self.tab_consistencia, text="Mercado (Over/Under e Handicaps)", padding=10)
        market_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.nb_market = ttk.Notebook(market_frame)
        self.nb_market.pack(fill="both", expand=True)

        self.tab_market_kills = ttk.Frame(self.nb_market)
        self.tab_market_towers = ttk.Frame(self.nb_market)
        self.tab_market_roshans = ttk.Frame(self.nb_market)
        self.tab_market_barracks = ttk.Frame(self.nb_market)
        self.tab_market_time = ttk.Frame(self.nb_market)
        self.tab_market_hc = ttk.Frame(self.nb_market)

        self.nb_market.add(self.tab_market_kills, text="Kills")
        self.nb_market.add(self.tab_market_towers, text="Torres")
        self.nb_market.add(self.tab_market_roshans, text="Roshans")
        self.nb_market.add(self.tab_market_barracks, text="Barracks")
        self.nb_market.add(self.tab_market_time, text="Tempo")
        self.nb_market.add(self.tab_market_hc, text="Handicaps")

        def _make_market_tree(parent):
            cols = ("Análise", "Ano Todo", "10 jogos", "5 jogos")
            tree = ttk.Treeview(parent, columns=cols, show="headings")
            tree.heading("Análise", text="Análise")
            tree.heading("Ano Todo", text="Ano Todo")
            tree.heading("10 jogos", text="10 jogos")
            tree.heading("5 jogos", text="5 jogos")

            tree.column("Análise", width=520, anchor="w")
            tree.column("Ano Todo", width=220, anchor="w")
            tree.column("10 jogos", width=220, anchor="w")
            tree.column("5 jogos", width=220, anchor="w")

            vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)

            tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")
            return tree

        self.market_trees = {
            "kills": _make_market_tree(self.tab_market_kills),
            "towers": _make_market_tree(self.tab_market_towers),
            "roshans": _make_market_tree(self.tab_market_roshans),
            "barracks": _make_market_tree(self.tab_market_barracks),
            "time": _make_market_tree(self.tab_market_time),
            "hc": _make_market_tree(self.tab_market_hc),
        }

        # mensagem inicial
        self._set_market_empty_message()

        # QoL: Enter = analisar
        self.root.bind("<Return>", lambda e: self.on_analisar())



    def _build_tab_prev_series(self):
        frm = ttk.Frame(self.tab_prev_series, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Previsão de Séries", padding=10)
        top.pack(side="top", fill="x", padx=10, pady=(0, 10))

        ttk.Label(
            top,
            text="Usa a mesma seleção (Liga/Time 1/Time 2) da aba 'Análise de Confronto'.\n"
                 "Clique em 'Calcular' para estimar P(mapa) e converter para BO1/BO3/BO5 + odds justas.",
            justify="left"
        ).grid(row=0, column=0, columnspan=8, sticky="w", padx=6, pady=(0, 10))

        # --------- Controles ----------
        ttk.Label(top, text="Formato:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.cb_series_tipo = ttk.Combobox(top, state="readonly", width=10, values=["BO1", "BO3", "BO5"])
        self.cb_series_tipo.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        self.cb_series_tipo.set("BO3")

        ttk.Label(top, text="Modelo:").grid(row=1, column=2, sticky="w", padx=(18, 6), pady=4)
        self.cb_series_modelo = ttk.Combobox(
            top, state="readonly", width=26,
            values=[
                "Híbrido (H2H + Liga)",
                "Somente H2H",
                "Somente Liga"
            ]
        )
        self.cb_series_modelo.grid(row=1, column=3, sticky="w", padx=6, pady=4)
        self.cb_series_modelo.set("Híbrido (H2H + Liga)")

        ttk.Label(top, text="Peso H2H:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.var_w_h2h = tk.StringVar(value="1.0")
        self.ent_w_h2h = ttk.Entry(top, textvariable=self.var_w_h2h, width=10)
        self.ent_w_h2h.grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(top, text="Peso Liga:").grid(row=2, column=2, sticky="w", padx=(18, 6), pady=4)
        self.var_w_liga = tk.StringVar(value="1.0")
        self.ent_w_liga = ttk.Entry(top, textvariable=self.var_w_liga, width=10)
        self.ent_w_liga.grid(row=2, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(top, text="Odds book (T1):").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.var_odds_t1 = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_odds_t1, width=10).grid(row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(top, text="Odds book (T2):").grid(row=3, column=2, sticky="w", padx=(18, 6), pady=4)
        self.var_odds_t2 = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_odds_t2, width=10).grid(row=3, column=3, sticky="w", padx=6, pady=4)

        self.var_series_info = tk.StringVar(value="Pronto para calcular.")
        ttk.Label(top, textvariable=self.var_series_info, foreground="#444").grid(
            row=1, column=4, rowspan=3, columnspan=4, sticky="w", padx=(18, 6), pady=4
        )

        def _safe_float(s: str, default: Optional[float] = None) -> Optional[float]:
            try:
                if s is None:
                    return default
                s = str(s).strip().replace(",", ".")
                if s == "":
                    return default
                return float(s)
            except Exception:
                return default

        def _toggle_weights(*_):
            mode = self.cb_series_modelo.get()
            if mode == "Híbrido (H2H + Liga)":
                self.ent_w_h2h.configure(state="normal")
                self.ent_w_liga.configure(state="normal")
            elif mode == "Somente H2H":
                self.ent_w_h2h.configure(state="normal")
                self.ent_w_liga.configure(state="disabled")
            else:
                self.ent_w_h2h.configure(state="disabled")
                self.ent_w_liga.configure(state="normal")

        self.cb_series_modelo.bind("<<ComboboxSelected>>", _toggle_weights)
        _toggle_weights()

        # --------- Saídas ----------
        out = ttk.Frame(frm)
        out.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        base_box = ttk.LabelFrame(out, text="Base (P(mapa))", padding=10)
        base_box.pack(side="top", fill="x", pady=(0, 10))

        self.tree_series_base = ttk.Treeview(base_box, columns=("Fonte", "Jogos", "Vit_T1", "P_T1"), show="headings", height=4)
        for col, w in [("Fonte", 180), ("Jogos", 90), ("Vit_T1", 90), ("P_T1", 120)]:
            self.tree_series_base.heading(col, text=col)
            self.tree_series_base.column(col, width=w, anchor="center")
        self.tree_series_base.pack(fill="x", expand=True)

        mk_box = ttk.LabelFrame(out, text="Resultado (odds justas + EV se preencher odds book)", padding=10)
        mk_box.pack(side="top", fill="both", expand=True, pady=(0, 10))

        self.tree_series_markets = ttk.Treeview(
            mk_box,
            columns=("Mercado", "Prob", "Odd_justa", "Odd_book", "EV"),
            show="headings", height=10
        )
        for col, w, anc in [
            ("Mercado", 320, "w"),
            ("Prob", 90, "center"),
            ("Odd_justa", 110, "center"),
            ("Odd_book", 110, "center"),
            ("EV", 90, "center"),
        ]:
            self.tree_series_markets.heading(col, text=col)
            self.tree_series_markets.column(col, width=w, anchor=anc)
        self.tree_series_markets.pack(fill="both", expand=True)

        score_box = ttk.LabelFrame(out, text="Placares Exatos", padding=10)
        score_box.pack(side="top", fill="x")

        self.tree_series_scores = ttk.Treeview(
            score_box, columns=("Placar", "Prob", "Odd_justa"), show="headings", height=6
        )
        for col, w in [("Placar", 220), ("Prob", 90), ("Odd_justa", 120)]:
            self.tree_series_scores.heading(col, text=col)
            self.tree_series_scores.column(col, width=w, anchor="center")
        self.tree_series_scores.pack(fill="x", expand=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", padx=10, pady=(0, 10))

        def _clear_tree(tree: ttk.Treeview):
            for iid in tree.get_children():
                tree.delete(iid)

        def _get_team_rows(df_scope: pd.DataFrame, team: str, opp: Optional[str] = None) -> pd.DataFrame:
            if df_scope is None or df_scope.empty:
                return pd.DataFrame()
            t = norm_team(team)
            out_df = df_scope[df_scope["teamname_norm"] == t].copy()
            if opp is not None:
                out_df = out_df[out_df["opp_teamname_norm"] == norm_team(opp)].copy()
            # 1 linha por match
            if "match_id" in out_df.columns:
                out_df = out_df.sort_values("date", ascending=False).drop_duplicates(subset=["match_id"])
            return out_df

        def _winrate_from_rows(rows: pd.DataFrame, alpha: float) -> Tuple[Optional[float], int, int]:
            if rows is None or rows.empty or "result" not in rows.columns:
                return None, 0, 0
            games = int(rows["match_id"].nunique()) if "match_id" in rows.columns else int(len(rows))
            wins = int(pd.to_numeric(rows["result"], errors="coerce").fillna(0).astype(float).sum())
            if games <= 0:
                return None, 0, 0
            p = (wins + alpha) / (games + 2.0 * alpha) if alpha > 0 else (wins / games)
            return float(p), games, wins

        def _p_series_from_pmap(p: float, bo: int) -> float:
            # Best-of bo: vence quem pega ceil(bo/2)
            need = (bo // 2) + 1
            out_p = 0.0
            q = 1.0 - p
            for k in range(need, bo + 1):
                out_p += math.comb(bo, k) * (p ** k) * (q ** (bo - k))
            return float(out_p)

        def _exact_score_probs(p: float, bo: int) -> List[Tuple[str, float]]:
            # Retorna placares do ponto de vista do T1.
            # Fórmula de stopping time: P(T1 m-n) = C(m+n-1, m-1) p^m (1-p)^n, onde m = wins necessários.
            need = (bo // 2) + 1
            q = 1.0 - p
            out = []
            # vitórias do T1 (m-n)
            for n_losses in range(0, need):
                ways = math.comb((need + n_losses - 1), (need - 1))
                prob = ways * (p ** need) * (q ** n_losses)
                out.append((f"{need}-{n_losses}", float(prob)))
            # derrotas do T1 (n-m)
            for n_wins in range(0, need):
                ways = math.comb((need + n_wins - 1), (need - 1))
                prob = ways * (q ** need) * (p ** n_wins)
                out.append((f"{n_wins}-{need}", float(prob)))
            # ordenar por mais provável
            out = sorted(out, key=lambda x: x[1], reverse=True)
            return out

        def _handicap_markets(p: float, bo: int) -> List[Tuple[str, float]]:
            # mercados mais comuns (map handicap)
            q = 1.0 - p
            need = (bo // 2) + 1

            markets = []
            # ML série
            markets.append(("ML Série (T1)", _p_series_from_pmap(p, bo)))
            markets.append(("ML Série (T2)", _p_series_from_pmap(q, bo)))

            if bo == 1:
                return markets

            if bo == 3:
                # T1 -1.5 => 2-0
                markets.append(("Handicap Mapas T1 -1.5 (2-0)", p ** 2))
                markets.append(("Handicap Mapas T2 -1.5 (0-2)", q ** 2))
                # +1.5 => ganha pelo menos 1 mapa
                markets.append(("Handicap Mapas T1 +1.5 (>=1 mapa)", 1.0 - (q ** 2)))
                markets.append(("Handicap Mapas T2 +1.5 (>=1 mapa)", 1.0 - (p ** 2)))
                return markets

            if bo == 5:
                # T1 -2.5 => 3-0
                markets.append(("Handicap Mapas T1 -2.5 (3-0)", p ** 3))
                markets.append(("Handicap Mapas T2 -2.5 (0-3)", q ** 3))
                # T1 -1.5 => 3-0 ou 3-1
                markets.append(("Handicap Mapas T1 -1.5 (3-0/3-1)", (p ** 3) + 3 * (p ** 3) * q))
                markets.append(("Handicap Mapas T2 -1.5 (0-3/1-3)", (q ** 3) + 3 * (q ** 3) * p))
                # +1.5 => perde no máximo por 1 mapa (ou vence)
                markets.append(("Handicap Mapas T1 +1.5 (>=2 mapas)", 1.0 - ((q ** 3) + 3 * (q ** 3) * p)))
                markets.append(("Handicap Mapas T2 +1.5 (>=2 mapas)", 1.0 - ((p ** 3) + 3 * (p ** 3) * q)))
                # +2.5 => ganha pelo menos 1 mapa
                markets.append(("Handicap Mapas T1 +2.5 (>=1 mapa)", 1.0 - (q ** 3)))
                markets.append(("Handicap Mapas T2 +2.5 (>=1 mapa)", 1.0 - (p ** 3)))
                return markets

            return markets

        def calcular():
            if not self.ds:
                messagebox.showerror("Erro", "Dataset não carregado.")
                return

            league, t1, t2 = self._get_selection()
            current_teams = self._combo_full_values.get(self.combo_t1, []) or []
            t1_res = self._resolve_from_list(t1, current_teams)
            t2_res = self._resolve_from_list(t2, current_teams)
            t1 = t1_res if t1_res else t1
            t2 = t2_res if t2_res else t2

            if not t1 or not t2:
                messagebox.showwarning("Atenção", "Selecione Time 1 e Time 2 (na aba Análise de Confronto).")
                return
            if norm_team(t1) == norm_team(t2):
                messagebox.showwarning("Atenção", "Selecione dois times diferentes.")
                return

            # Alpha Laplace (liga ao checkbox da UI)
            alpha = 1.0 if bool(self.laplace_var.get()) else 0.0

            df_scope = self.ds.filter_league(league)
            # h2h (t1 como referência)
            rows_h2h = _get_team_rows(df_scope, t1, t2)
            p_h2h, g_h2h, w_h2h = _winrate_from_rows(rows_h2h, alpha)

            # liga (exclui os match_ids h2h do sample da liga)
            match_ids_h2h = set(rows_h2h["match_id"].unique()) if (rows_h2h is not None and not rows_h2h.empty and "match_id" in rows_h2h.columns) else set()
            rows_liga = _get_team_rows(df_scope, t1, None)
            if match_ids_h2h and rows_liga is not None and not rows_liga.empty and "match_id" in rows_liga.columns:
                rows_liga = rows_liga[~rows_liga["match_id"].isin(match_ids_h2h)].copy()
            p_liga, g_liga, w_liga = _winrate_from_rows(rows_liga, alpha)

            # escolher modelo
            mode = self.cb_series_modelo.get()
            p_map = None

            if mode == "Somente H2H":
                p_map = p_h2h
            elif mode == "Somente Liga":
                p_map = p_liga
            else:
                wh = _safe_float(self.var_w_h2h.get(), 1.0) or 0.0
                wl = _safe_float(self.var_w_liga.get(), 1.0) or 0.0

                parts = []
                if p_h2h is not None and wh > 0:
                    parts.append((p_h2h, wh))
                if p_liga is not None and wl > 0:
                    parts.append((p_liga, wl))

                if parts:
                    num = sum(pv * w for pv, w in parts)
                    den = sum(w for _, w in parts)
                    p_map = (num / den) if den > 0 else None

            if p_map is None:
                messagebox.showwarning(
                    "Sem dados",
                    "Não encontrei jogos suficientes para calcular P(mapa).\n"
                    "Teste: mudar o Modelo para 'Somente Liga' ou 'Somente H2H', ou usar '(Todas)' na liga."
                )
                return

            # evita odds infinitas
            p_map = max(0.01, min(0.99, float(p_map)))

            bo = int(self.cb_series_tipo.get().replace("BO", "").strip())
            p_series = _p_series_from_pmap(p_map, bo)

            # atualizar base table
            _clear_tree(self.tree_series_base)
            if p_h2h is not None:
                self.tree_series_base.insert("", "end", values=("H2H", g_h2h, w_h2h, f"{p_h2h:.4f}"))
            else:
                self.tree_series_base.insert("", "end", values=("H2H", 0, 0, "-"))
            if p_liga is not None:
                self.tree_series_base.insert("", "end", values=("Liga (sem H2H)", g_liga, w_liga, f"{p_liga:.4f}"))
            else:
                self.tree_series_base.insert("", "end", values=("Liga (sem H2H)", 0, 0, "-"))
            self.tree_series_base.insert("", "end", values=("P(mapa) usado", "-", "-", f"{p_map:.4f}"))

            # mercados
            _clear_tree(self.tree_series_markets)
            odds_book_t1 = _safe_float(self.var_odds_t1.get(), None)
            odds_book_t2 = _safe_float(self.var_odds_t2.get(), None)

            def _row(mkt: str, prob: float, odd_book: Optional[float] = None):
                prob = max(0.000001, min(0.999999, float(prob)))
                odd_fair = 1.0 / prob
                ev = ""
                ob = ""
                if odd_book is not None and odd_book > 0:
                    ob = f"{odd_book:.2f}"
                    ev_val = prob * odd_book - 1.0
                    ev = f"{ev_val:+.3f}"
                self.tree_series_markets.insert(
                    "", "end",
                    values=(mkt, f"{prob:.4f}", f"{odd_fair:.2f}", ob, ev)
                )

            # linha "ML série" leva odds book
            _row(f"ML Série - {t1}", p_series, odds_book_t1)
            _row(f"ML Série - {t2}", 1.0 - p_series, odds_book_t2)

            for mkt, prob in _handicap_markets(p_map, bo):
                if "ML Série" in mkt:
                    continue
                _row(mkt.replace("T1", t1).replace("T2", t2), prob, None)

            # placar exato
            _clear_tree(self.tree_series_scores)
            for placar, prob in _exact_score_probs(p_map, bo):
                prob = max(0.000001, min(0.999999, float(prob)))
                odd = 1.0 / prob
                self.tree_series_scores.insert("", "end", values=(f"{t1} {placar} {t2}", f"{prob:.4f}", f"{odd:.2f}"))

            self.var_series_info.set(
                f"Liga: {league} | Formato: BO{bo} | Laplace: {'ON' if alpha>0 else 'OFF'} | "
                f"P(mapa)={p_map:.4f} → P(série {t1})={p_series:.4f}"
            )

        ttk.Button(btns, text="Calcular Previsão", command=calcular).pack(side="left")
        ttk.Label(btns, text="Dica: se o H2H estiver vazio, use 'Somente Liga' ou selecione '(Todas)'.").pack(
            side="left", padx=(12, 0)
        )

    def _set_market_empty_message(self):
        # Limpa e coloca mensagem padrão em todas as abas do mercado
        if not hasattr(self, "market_trees"):
            return
        for _, tree in self.market_trees.items():
            for item in tree.get_children():
                tree.delete(item)
            tree.insert(
                "",
                "end",
                values=("Defina linhas em 'Adicionar/Mudar Linhas' para ver Over/Under e Handicap.", "", "", "")
            )


    def _fill_market_tabs(self, df_out: "pd.DataFrame"):
        """
        Preenche as abas do mercado separando por categoria:
        - Kills / Torres / Roshans / Barracks / Tempo -> O/U
        - Handicaps -> HC
        """
        if not hasattr(self, "market_trees"):
            return
        # limpa abas
        for tree in self.market_trees.values():
            for item in tree.get_children():
                tree.delete(item)

        def add_row(tree_key: str, row):
            tree = self.market_trees.get(tree_key)
            if not tree:
                return
            tree.insert("", "end", values=(row.get("Análise", ""), row.get("Ano Todo", ""), row.get("10 jogos", ""), row.get("5 jogos", "")))

        if df_out is None or df_out.empty:
            self._set_market_empty_message()
            return

        for _, r in df_out.iterrows():
            name = str(r.get("Análise", "") or "")
            row = {
                "Análise": name,
                "Ano Todo": r.get("Ano Todo", ""),
                "10 jogos": r.get("10 jogos", ""),
                "5 jogos": r.get("5 jogos", ""),
            }

            up = name.upper()

            # Handicaps (qualquer linha com " HC ")
            if up.startswith("HC ") or " HC " in up:
                add_row("hc", row)
            # Tempo
            elif "TEMPO" in up:
                add_row("time", row)
            # Mercados
            elif "KILLS" in up:
                add_row("kills", row)
            elif "TORRES" in up or "TOWERS" in up:
                add_row("towers", row)
            elif "ROSHAN" in up:
                add_row("roshans", row)
            elif "BARRACK" in up or "BARRACKS" in up or "QUARTEL" in up:
                add_row("barracks", row)
            else:
                # fallback: joga em Kills pra não sumir
                add_row("kills", row)

        # Se alguma aba ficou vazia, coloca um aviso leve (não polui com 200 mensagens)
        for key, tree in self.market_trees.items():
            if len(tree.get_children()) == 0:
                tree.insert("", "end", values=("Sem dados para este mercado (com as linhas atuais).", "", "", ""))

        # Foca na primeira aba com dados (normalmente Kills)
        try:
            self.nb_market.select(self.tab_market_kills)
        except Exception:
            pass


    def _fill_overview_tables(self, overview: Dict[str, dict], team1: str, team2: str):
        """
        Preenche as tabelas da aba "Visão Geral" (Ano Todo / Últimos 10 / Últimos 5)
        usando o dict retornado por DotaAnalyzer.compute_overview.

        overview = {
            "Ano Todo": {"h2h": {...}, "t1": {...}, "t2": {...}, "all": {...}},
            "10 jogos": {...},
            "5 jogos": {...},
        }
        """
        if not hasattr(self, "_overview_trees"):
            return

        def fmt_num(x, dec=1):
            try:
                if x is None:
                    return "-"
                if isinstance(x, float) and (x != x):  # NaN
                    return "-"
                return f"{float(x):.{dec}f}"
            except Exception:
                return "-"

        for period_label, key in [("Ano Todo", "year"), ("10 jogos", "last10"), ("5 jogos", "last5")]:
            trees = self._overview_trees.get(key)
            if not trees:
                continue
            tree_h2h, tree_cmp = trees

            # limpa
            for t in (tree_h2h, tree_cmp):
                for it in t.get_children():
                    t.delete(it)

            data_p = overview.get(period_label) or {}
            h2h = data_p.get("h2h") or {}
            t1 = data_p.get("t1") or {}
            t2 = data_p.get("t2") or {}
            allp = data_p.get("all") or {}

            # ----- H2H -----
            n_h = int(h2h.get("games", 0) or 0)
            if n_h <= 0:
                tree_h2h.insert("", "end", values=("Sem jogos H2H neste período", ""))
            else:
                rows_h = [
                    ("Total de jogos", str(n_h)),
                    ("Média de kills", fmt_num(h2h.get("avg_kills"), 1)),
                    ("Média de torres", fmt_num(h2h.get("avg_towers"), 1)),
                    ("Média de roshans", fmt_num(h2h.get("avg_roshans"), 2)),
                    ("Média de barracks", fmt_num(h2h.get("avg_barracks"), 2)),
                    ("Média de tempo", fmt_mmss_from_min(h2h.get("avg_time"))),
                ]
                for met, val in rows_h:
                    tree_h2h.insert("", "end", values=(met, val))

            # ----- Comparativo Liga / Escopo -----
            n1 = int(t1.get("games", 0) or 0)
            n2 = int(t2.get("games", 0) or 0)
            nt = int(allp.get("games", 0) or 0)

            # Atualiza cabeçalhos com nomes dos times
            try:
                tree_cmp.heading("T1", text=team1)
                tree_cmp.heading("T2", text=team2)
                tree_cmp.heading("MÉDIA GERAL", text="Média Geral")
            except Exception:
                pass

            if n1 == 0 and n2 == 0 and nt == 0:
                tree_cmp.insert("", "end", values=("Sem dados neste período", "", "", ""))
                continue

            rows_cmp = [
                ("Total de jogos", str(n1), str(n2), str(nt)),
                ("Média de kills",
                 fmt_num(t1.get("avg_kills"), 1) if n1 > 0 else "-",
                 fmt_num(t2.get("avg_kills"), 1) if n2 > 0 else "-",
                 fmt_num(allp.get("avg_kills"), 1) if nt > 0 else "-"),
                ("Média de torres",
                 fmt_num(t1.get("avg_towers"), 1) if n1 > 0 else "-",
                 fmt_num(t2.get("avg_towers"), 1) if n2 > 0 else "-",
                 fmt_num(allp.get("avg_towers"), 1) if nt > 0 else "-"),
                ("Média de roshans",
                 fmt_num(t1.get("avg_roshans"), 2) if n1 > 0 else "-",
                 fmt_num(t2.get("avg_roshans"), 2) if n2 > 0 else "-",
                 fmt_num(allp.get("avg_roshans"), 2) if nt > 0 else "-"),
                ("Média de barracks",
                 fmt_num(t1.get("avg_barracks"), 2) if n1 > 0 else "-",
                 fmt_num(t2.get("avg_barracks"), 2) if n2 > 0 else "-",
                 fmt_num(allp.get("avg_barracks"), 2) if nt > 0 else "-"),
                ("Média de tempo",
                 fmt_mmss_from_min(t1.get("avg_time")) if n1 > 0 else "-",
                 fmt_mmss_from_min(t2.get("avg_time")) if n2 > 0 else "-",
                 fmt_mmss_from_min(allp.get("avg_time")) if nt > 0 else "-"),
            ]

            for row in rows_cmp:
                tree_cmp.insert("", "end", values=row)

    def _populate_lists(self):
        if not self.ds:
            return
        ligas = self.ds.leagues()
        self._set_combo_values(self.combo_liga, ligas)
        self.combo_liga.current(0)

        self._update_teams()
        self._log(f"Ligas carregadas: {len(ligas)-1} (+ Todas)\n")

    def _update_teams(self):
        if not self.ds:
            return
        league = (self.combo_liga.get() or "(Todas)").strip()
        # Se o usuário digitou uma liga parcial/sem acento, tenta resolver.
        full_leagues = self._combo_full_values.get(self.combo_liga, []) or self.ds.leagues()
        if league and league != "(Todas)" and league not in full_leagues:
            resolved = self._resolve_from_list(league, full_leagues)
            if resolved in full_leagues:
                league = resolved
                self.combo_liga.set(league)
            else:
                league = "(Todas)"
                self.combo_liga.set(league)
        teams = self.ds.teams_for_league(league)
        self._set_combo_values(self.combo_t1, teams)
        self._set_combo_values(self.combo_t2, teams)
        self._log(f"Atualizado times para liga '{league}' ({len(teams)} times)\n")

    def _get_selection(self) -> Tuple[str, str, str]:
        league = self.combo_liga.get() or "(Todas)"
        t1 = self.combo_t1.get().strip()
        t2 = self.combo_t2.get().strip()
        return league, t1, t2

    def on_linhas(self):
        # abre janela de linhas
        def apply(cfg: LinesConfig):
            self.lines_cfg = cfg
            self._log("Linhas atualizadas.\n")
            # QoL: recalcula automaticamente se já tiver times selecionados
            league, t1, t2 = self._get_selection()
            if t1 and t2:
                self.on_analisar()

        LinesWindow(self.root, self.lines_cfg, apply)


    def _update_overview(self, league: str, t1: str, t2: str):
        """
        Atualiza as tabelas de Visão Geral (H2H + Liga) com as médias dos dois times.
        """
        if not self.ds or not self.analyzer:
            return

        try:
            overview_h2h, overview_liga = self.analyzer.compute_overview(league, t1, t2)
        except Exception as e:
            self._log(f"[Visão Geral] Erro: {e}\n")
            return

        # limpa
        if hasattr(self, "tree_over_h2h"):
            for it in self.tree_over_h2h.get_children():
                self.tree_over_h2h.delete(it)
        if hasattr(self, "tree_over_liga"):
            for it in self.tree_over_liga.get_children():
                self.tree_over_liga.delete(it)

        # H2H
        if hasattr(self, "tree_over_h2h"):
            n = overview_h2h.get("games", 0) or 0
            if n <= 0:
                self.tree_over_h2h.insert("", "end", values=("Sem jogos H2H", ""))
            else:
                rows = [
                    ("Total de jogos", str(n)),
                    ("Média de kills", f"{overview_h2h.get('kills', float('nan')):.1f}"),
                    ("Média de torres", f"{overview_h2h.get('towers', float('nan')):.1f}"),
                    ("Média de roshans", f"{overview_h2h.get('roshans', float('nan')):.2f}"),
                    ("Média de barracks", f"{overview_h2h.get('barracks', float('nan')):.2f}"),
                    ("Média de tempo", fmt_mmss_from_min(overview_h2h.get('time_min', float('nan')))),
                ]
                for r in rows:
                    self.tree_over_h2h.insert("", "end", values=r)

        # Liga
        if hasattr(self, "tree_over_liga"):
            n1 = overview_liga.get("games_t1", 0) or 0
            n2 = overview_liga.get("games_t2", 0) or 0
            total = overview_liga.get("games_total", 0) or 0

            label_t1 = f"{t1} (Ano)"
            label_t2 = f"{t2} (Ano)"
            label_all = "Média Geral"

            rows_l = [
                ("Total de jogos", str(n1), str(n2), str(total)),
                ("Média de kills",
                 f"{overview_liga.get('kills_t1', float('nan')):.1f}" if n1 > 0 else "-",
                 f"{overview_liga.get('kills_t2', float('nan')):.1f}" if n2 > 0 else "-",
                 f"{overview_liga.get('kills_all', float('nan')):.1f}" if total > 0 else "-"),
                ("Média de torres",
                 f"{overview_liga.get('towers_t1', float('nan')):.1f}" if n1 > 0 else "-",
                 f"{overview_liga.get('towers_t2', float('nan')):.1f}" if n2 > 0 else "-",
                 f"{overview_liga.get('towers_all', float('nan')):.1f}" if total > 0 else "-"),
                ("Média de roshans",
                 f"{overview_liga.get('roshans_t1', float('nan')):.2f}" if n1 > 0 else "-",
                 f"{overview_liga.get('roshans_t2', float('nan')):.2f}" if n2 > 0 else "-",
                 f"{overview_liga.get('roshans_all', float('nan')):.2f}" if total > 0 else "-"),
                ("Média de barracks",
                 f"{overview_liga.get('barracks_t1', float('nan')):.2f}" if n1 > 0 else "-",
                 f"{overview_liga.get('barracks_t2', float('nan')):.2f}" if n2 > 0 else "-",
                 f"{overview_liga.get('barracks_all', float('nan')):.2f}" if total > 0 else "-"),
                ("Média de tempo",
                 fmt_mmss_from_min(overview_liga.get('time_t1', float('nan'))) if n1 > 0 else "-",
                 fmt_mmss_from_min(overview_liga.get('time_t2', float('nan'))) if n2 > 0 else "-",
                 fmt_mmss_from_min(overview_liga.get('time_all', float('nan'))) if total > 0 else "-"),
            ]

            # Ajusta header com nome dos times
            self.tree_over_liga.heading("t1", text=label_t1)
            self.tree_over_liga.heading("t2", text=label_t2)
            self.tree_over_liga.heading("geral", text=label_all)

            for lbl, v1, v2, vg in rows_l:
                self.tree_over_liga.insert("", "end", values=(lbl, v1, v2, vg))

    def on_analisar(self):
        if not self.ds or not self.analyzer:
            messagebox.showerror("Erro", "Dataset não carregado.")
            return

        league, t1, t2 = self._get_selection()
        # QoL: se o usuário digitou parcialmente, tenta resolver para um time real.
        current_teams = self._combo_full_values.get(self.combo_t1, []) or []

        # tenta resolver melhor (match exato / único)
        t1_res = self._resolve_from_list(t1, current_teams)
        t2_res = self._resolve_from_list(t2, current_teams)

        # garante que os valores sejam strings simples para o Combobox
        if isinstance(t1_res, (list, tuple)):
            t1_res = t1_res[0] if t1_res else ""
        if isinstance(t2_res, (list, tuple)):
            t2_res = t2_res[0] if t2_res else ""
        t1_res = "" if t1_res is None else str(t1_res)
        t2_res = "" if t2_res is None else str(t2_res)

        if t1_res and t1_res != t1:
            self.combo_t1.set(t1_res)
            t1 = t1_res
        if t2_res and t2_res != t2:
            self.combo_t2.set(t2_res)
            t2 = t2_res

        # se ainda não bateu, monta lista de sugestões (contém)
        def suggestions(typed: str) -> List[str]:
            needle = self._norm_simple(typed)
            if not needle:
                return []
            return [v for v in current_teams if needle in self._norm_simple(v)]

        if (t1 and current_teams) and (t1 not in current_teams):
            sug = suggestions(t1)
            if len(sug) == 1:
                self.combo_t1.set(sug[0]); t1 = sug[0]
            elif len(sug) > 1:
                choice = self._choose_from_list("Escolher Time 1", t1, sug[:500])
                if not choice:
                    return
                self.combo_t1.set(choice); t1 = choice
            else:
                messagebox.showwarning("Atenção", "Time 1 não encontrado. Digite outra sequência ou selecione no dropdown.")
                return

        if (t2 and current_teams) and (t2 not in current_teams):
            sug = suggestions(t2)
            if len(sug) == 1:
                self.combo_t2.set(sug[0]); t2 = sug[0]
            elif len(sug) > 1:
                choice = self._choose_from_list("Escolher Time 2", t2, sug[:500])
                if not choice:
                    return
                self.combo_t2.set(choice); t2 = choice
            else:
                messagebox.showwarning("Atenção", "Time 2 não encontrado. Digite outra sequência ou selecione no dropdown.")
                return
        if not t1 or not t2:
            messagebox.showwarning("Atenção", "Selecione Time 1 e Time 2.")
            return
        if norm_team(t1) == norm_team(t2):
            messagebox.showwarning("Atenção", "Selecione dois times diferentes.")
            return

        self._log(f"Analisando: {t1} x {t2} | Liga: {league}\n")

        # 1) Sempre atualiza a Visão Geral (médias/quantidade de jogos)
        try:
            overview = self.analyzer.compute_overview(league=league, team1=t1, team2=t2)
            self._fill_overview_tables(overview, t1, t2)
        except Exception as e:
            self._log(f"Falha ao montar Visão Geral: {e}\n")

        # 2) Mercado (Over/Under e Handicaps) só aparece se houver linhas
        if self.lines_cfg.is_empty():
            self._set_market_empty_message()
            # foca na Visão Geral
            try:
                self.nb_confronto.select(self.tab_visao_geral)
            except Exception:
                pass
            return

        try:
            df_out = self.analyzer.analyze(
                league=league,
                team1=t1,
                team2=t2,
                lines=self.lines_cfg,
                laplace=bool(self.laplace_var.get()),
            )
            self._fill_market_tabs(df_out)
        except Exception as e:
            messagebox.showerror("Erro ao analisar", str(e))
            self._log(f"Erro ao analisar: {e}\n")
            return

        # (tabela do mercado agora é preenchida por _fill_market_tabs)

        try:
            self.nb_confronto.select(self.tab_consistencia)
        except Exception:
            pass

    def on_resumo(self):
        # resumo textual rápido (QoL)
        league, t1, t2 = self._get_selection()
        msg = [
            f"Resumo (Dota): {t1} x {t2}",
            f"Liga: {league}",
            "",
            f"Linhas Totais:",
            f"  Kills: {self.lines_cfg.total_kills}",
            f"  Torres: {self.lines_cfg.total_towers}",
            f"  Roshans: {self.lines_cfg.total_roshans}",
            f"  Barracks: {self.lines_cfg.total_barracks}",
            f"  Tempo (min): {self.lines_cfg.total_time_min}",
            "",
            f"Handicaps (T1):",
            f"  Kills: {self.lines_cfg.hc_kills_t1}",
            f"  Torres: {self.lines_cfg.hc_towers_t1}",
            f"  Roshans: {self.lines_cfg.hc_roshans_t1}",
            f"  Barracks: {self.lines_cfg.hc_barracks_t1}",
        ]
        messagebox.showinfo("Resumo", "\n".join(msg))


def main():
    if tk is None:
        raise RuntimeError(
            "Tkinter não está disponível neste ambiente. "
            "Para rodar no navegador, use o Streamlit (arquivo streamlit_app.py)."
        )
    root = tk.Tk()
    app = GabineteDotaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
