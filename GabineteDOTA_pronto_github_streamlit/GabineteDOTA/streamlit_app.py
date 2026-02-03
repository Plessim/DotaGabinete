# -*- coding: utf-8 -*-
"""
streamlit_app.py
----------------
Versão WEB (Streamlit) do Gabinete Dota.

Observação:
- O arquivo GabineteDotaFINAL.py (Tkinter) foi mantido para uso desktop.
- Em ambientes "headless" (ex.: Streamlit Community Cloud), Tkinter pode não existir.
  Por isso, a interface WEB roda por este arquivo.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

from GabineteDotaFINAL import (
    DotaDataset,
    DotaAnalyzer,
    LinesConfig,
    parse_float_list,
    parse_time_line,
    norm_team,
)

APP_TITLE = "Gabinete Dota (Web)"
SAVED_GAMES_FILE = "saved_games_dota.json"


# -------------------------
# Helpers (saved games)
# -------------------------
def _saved_file_path() -> Path:
    return Path(__file__).resolve().parent / SAVED_GAMES_FILE


def _make_game_key(league: str, t1: str, t2: str) -> str:
    ln = norm_team(league or "")
    a = norm_team(t1 or "")
    b = norm_team(t2 or "")
    pair = "::".join(sorted([a, b]))
    return f"{ln}::{pair}"


def load_saved_games() -> Dict[str, Dict[str, Any]]:
    """Carrega saved_games_dota.json (se existir) e retorna dict key->item."""
    out: Dict[str, Dict[str, Any]] = {}
    fp = _saved_file_path()
    if not fp.exists():
        return out
    try:
        data = json.loads(fp.read_text(encoding="utf-8")) or []
        if isinstance(data, dict):
            data = list(data.values())
        for item in data:
            key = item.get("key")
            if key:
                out[key] = item
    except Exception:
        # Não quebra o app caso o arquivo esteja corrompido
        return {}
    return out


def persist_saved_games(saved: Dict[str, Dict[str, Any]]) -> bool:
    """Tenta persistir no arquivo local. Em Streamlit Cloud pode não ser permanente."""
    fp = _saved_file_path()
    try:
        fp.write_text(json.dumps(list(saved.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def lines_to_text(lines: LinesConfig) -> Dict[str, str]:
    """Para preencher as caixas de texto."""
    def join_nums(xs: List[float]) -> str:
        return ", ".join([str(x).rstrip("0").rstrip(".") if isinstance(x, float) else str(x) for x in xs])

    return {
        "total_kills_text": join_nums(lines.total_kills),
        "total_towers_text": join_nums(lines.total_towers),
        "total_roshans_text": join_nums(lines.total_roshans),
        "total_barracks_text": join_nums(lines.total_barracks),
        "total_time_text": join_nums(lines.total_time_min),
        "hc_kills_text": join_nums(lines.hc_kills_t1),
        "hc_towers_text": join_nums(lines.hc_towers_t1),
        "hc_roshans_text": join_nums(lines.hc_roshans_t1),
        "hc_barracks_text": join_nums(lines.hc_barracks_t1),
    }


def build_lines_from_state() -> LinesConfig:
    return LinesConfig(
        total_kills=parse_float_list(st.session_state.get("total_kills_text", "")),
        total_towers=parse_float_list(st.session_state.get("total_towers_text", "")),
        total_roshans=parse_float_list(st.session_state.get("total_roshans_text", "")),
        total_barracks=parse_float_list(st.session_state.get("total_barracks_text", "")),
        total_time_min=parse_time_line(st.session_state.get("total_time_text", "")),
        hc_kills_t1=parse_float_list(st.session_state.get("hc_kills_text", "")),
        hc_towers_t1=parse_float_list(st.session_state.get("hc_towers_text", "")),
        hc_roshans_t1=parse_float_list(st.session_state.get("hc_roshans_text", "")),
        hc_barracks_t1=parse_float_list(st.session_state.get("hc_barracks_text", "")),
    )


# -------------------------
# Caching
# -------------------------
@st.cache_data(show_spinner=False)
def _load_dataset() -> DotaDataset:
    return DotaDataset("data.csv")


@st.cache_resource(show_spinner=False)
def _load_analyzer() -> DotaAnalyzer:
    ds = _load_dataset()
    return DotaAnalyzer(ds)


# -------------------------
# App
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption("Versão web (Streamlit) baseada no Gabinete Dota. Use a aba “Salvar/Carregar” para guardar linhas e confrontos.")

# Init state
if "saved_games" not in st.session_state:
    st.session_state["saved_games"] = load_saved_games()

ds = _load_dataset()
analyzer = _load_analyzer()

# Defaults for selectors (before widgets)
league_options = ds.leagues()
if st.session_state.get("league") not in league_options:
    st.session_state["league"] = "(Todas)"

team_options = ds.teams_for_league(st.session_state["league"])
if team_options:
    if st.session_state.get("team1") not in team_options:
        st.session_state["team1"] = team_options[0]
    if st.session_state.get("team2") not in team_options or st.session_state.get("team2") == st.session_state.get("team1"):
        # tenta escolher diferente
        t2 = next((t for t in team_options if t != st.session_state["team1"]), team_options[0])
        st.session_state["team2"] = t2

tabs = st.tabs(["Análise", "Salvar/Carregar", "Dados"])

with tabs[0]:
    left, right = st.columns([1.1, 1.3], gap="large")

    with left:
        st.subheader("Seleção")
        st.selectbox("Liga", league_options, key="league")
        team_options = ds.teams_for_league(st.session_state["league"])
        st.selectbox("Time 1", team_options, key="team1")
        team2_options = [t for t in team_options if t != st.session_state.get("team1")] or team_options
        # Se o time2 atual não estiver nas opções (por mudança de liga), ajusta
        if st.session_state.get("team2") not in team2_options and team2_options:
            st.session_state["team2"] = team2_options[0]
        st.selectbox("Time 2", team2_options, key="team2")

        st.checkbox("Usar Laplace (recomendado)", value=True, key="use_laplace")

        st.divider()
        st.subheader("Linhas")

        with st.expander("Totais (Over/Under)", expanded=True):
            st.text_area("Kills (ex.: 46.5, 49.5)", key="total_kills_text", height=70)
            st.text_area("Torres (ex.: 12.5)", key="total_towers_text", height=70)
            st.text_area("Roshans (ex.: 2.5)", key="total_roshans_text", height=70)
            st.text_area("Barracks (ex.: 5.5)", key="total_barracks_text", height=70)
            st.text_area("Tempo (min) (ex.: 39, 39.5 ou 38:30)", key="total_time_text", height=70)

        with st.expander("Handicaps (apenas Time 1)", expanded=False):
            st.text_area("Kills (ex.: -2.5)", key="hc_kills_text", height=70)
            st.text_area("Torres", key="hc_towers_text", height=70)
            st.text_area("Roshans", key="hc_roshans_text", height=70)
            st.text_area("Barracks", key="hc_barracks_text", height=70)

        run = st.button("Analisar confronto", type="primary", use_container_width=True)

    with right:
        st.subheader("Resultado")
        if run:
            lines_cfg = build_lines_from_state()

            if lines_cfg.is_empty():
                st.warning("Defina pelo menos uma linha para calcular.")
            else:
                with st.spinner("Calculando…"):
                    df = analyzer.analyze(
                        st.session_state["league"],
                        st.session_state["team1"],
                        st.session_state["team2"],
                        lines_cfg,
                        laplace=bool(st.session_state.get("use_laplace", True)),
                    )

                st.dataframe(df, use_container_width=True, hide_index=True)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar resultado (CSV)",
                    data=csv_bytes,
                    file_name=f"analise_{norm_team(st.session_state['team1'])}_x_{norm_team(st.session_state['team2'])}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Selecione liga/times, defina as linhas e clique em **Analisar confronto**.")

with tabs[1]:
    st.subheader("Salvar confrontos e linhas")

    saved: Dict[str, Dict[str, Any]] = st.session_state["saved_games"]

    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        st.markdown("**Salvar o confronto atual**")
        lines_cfg = build_lines_from_state()
        if st.button("Salvar/Atualizar", use_container_width=True):
            key = _make_game_key(st.session_state.get("league", ""), st.session_state.get("team1", ""), st.session_state.get("team2", ""))
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            item = saved.get(key) or {}
            created_at = item.get("created_at") or now

            saved[key] = {
                "key": key,
                "league": st.session_state.get("league", "(Todas)"),
                "team1": st.session_state.get("team1", ""),
                "team2": st.session_state.get("team2", ""),
                "created_at": created_at,
                "updated_at": now,
                "lines": lines_cfg.to_dict(),
            }

            ok = persist_saved_games(saved)
            st.session_state["saved_games"] = saved
            if ok:
                st.success("Salvo! (Obs.: em hospedagens gratuitas, isso pode não persistir após reinícios do app.)")
            else:
                st.warning("Salvo na sessão, mas não consegui gravar no arquivo. Use o botão de exportar JSON para guardar.")

        st.divider()
        st.markdown("**Exportar / Importar**")
        export_json = json.dumps(list(saved.values()), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Exportar saved_games_dota.json",
            data=export_json,
            file_name="saved_games_dota.json",
            mime="application/json",
            use_container_width=True,
        )

        up = st.file_uploader("Importar saved_games_dota.json", type=["json"])
        if up is not None:
            try:
                incoming = json.loads(up.getvalue().decode("utf-8"))
                if isinstance(incoming, dict):
                    incoming = list(incoming.values())
                n_added = 0
                for item in incoming:
                    k = item.get("key")
                    if k:
                        saved[k] = item
                        n_added += 1
                st.session_state["saved_games"] = saved
                persist_saved_games(saved)
                st.success(f"Importado: {n_added} registros.")
            except Exception as e:
                st.error(f"Falha ao importar: {e}")

    with colB:
        st.markdown("**Carregar confronto salvo**")
        if not saved:
            st.info("Nenhum confronto salvo ainda.")
        else:
            # Ordena por updated_at desc
            def _sort_key(item: Dict[str, Any]) -> str:
                return item.get("updated_at") or item.get("created_at") or ""

            items = sorted(saved.values(), key=_sort_key, reverse=True)
            labels = []
            keys = []
            for it in items:
                keys.append(it["key"])
                labels.append(f"{it.get('team1','')} x {it.get('team2','')}  |  {it.get('league','(Todas)')}  |  {it.get('updated_at','')}")
            sel = st.selectbox("Escolha", options=list(range(len(keys))), format_func=lambda i: labels[i])
            chosen = items[int(sel)]

            st.code(json.dumps(chosen.get("lines", {}), ensure_ascii=False, indent=2), language="json")

            if st.button("Carregar para o formulário", use_container_width=True):
                st.session_state["league"] = chosen.get("league", "(Todas)")
                st.session_state["team1"] = chosen.get("team1", "")
                st.session_state["team2"] = chosen.get("team2", "")

                lines = LinesConfig.from_dict(chosen.get("lines", {}))
                text_map = lines_to_text(lines)
                for k, v in text_map.items():
                    st.session_state[k] = v
                st.success("Carregado! Volte para a aba “Análise” e clique em Analisar.")

with tabs[2]:
    st.subheader("Dados (data.csv)")
    st.write("Este app lê o arquivo **data.csv** que está no repositório.")
    with st.expander("Prévia do dataset"):
        try:
            df = ds.df.copy()
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)
            st.caption(f"Linhas: {len(df):,} | Colunas: {len(df.columns)}")
        except Exception as e:
            st.error(str(e))
