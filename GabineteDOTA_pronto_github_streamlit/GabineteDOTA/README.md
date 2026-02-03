# Gabinete Dota (Web)

Este repositório contém duas versões do app:

- **Web (Streamlit)**: `streamlit_app.py` (recomendado para hospedar no navegador)
- **Desktop (Tkinter)**: `GabineteDotaFINAL.py` (uso local)

O app lê o arquivo `data.csv` que está no repositório.

---

## Rodar localmente (Web)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Deploy no Streamlit Community Cloud (usar no navegador)

1. Suba este projeto no GitHub (repo público ou privado).
2. No Streamlit Community Cloud, clique em **New app** e selecione:
   - Repository / Branch
   - **Main file path:** `streamlit_app.py`
3. Deploy.

> Observação: em hospedagens gratuitas, salvar confrontos em `saved_games_dota.json` pode não persistir após reinícios do app.
> Use a aba **Salvar/Carregar** para exportar/importar o JSON quando precisar.

---

## Rodar localmente (Desktop)

```bash
python GabineteDotaFINAL.py
```

Se aparecer erro de Tkinter em Linux/headless, use a versão web (Streamlit).
