"""
ANEEL Tariff Data Loader

Downloads and processes homologated tariff data from ANEEL Open Data portal
for BESS behind-the-meter viability analysis.

Data source: https://dadosabertos.aneel.gov.br/dataset/tarifas-distribuidoras-energia-eletrica
"""

import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR / "tarifas-raw.csv"

ANEEL_URL = (
    "https://dadosabertos.aneel.gov.br/dataset/"
    "5a583f3e-1646-4f67-bf0f-69db4203e89e/resource/"
    "fcf2906c-7c32-4b9b-a637-054e7a5234f4/download/"
    "tarifas-homologadas-distribuidoras-energia-eletrica.csv"
)

GRUPO_A = ["A1", "A2", "A3", "A3a", "A4"]
MODALITIES = ["Azul", "Verde"]

# Normalize seasonal tariff posts (north region has wet/dry variants)
POST_MAP = {
    "Ponta": "Ponta",
    "Ponta seca": "Ponta",
    "Ponta úmida": "Ponta",
    "Fora ponta": "Fora ponta",
    "Fora ponta seca": "Fora ponta",
    "Fora ponta úmida": "Fora ponta",
}


def download_if_needed():
    """Download tariff CSV from ANEEL if not already cached locally."""
    if RAW_FILE.exists():
        return
    print("Downloading ANEEL tariff data (~78 MB)...")
    resp = requests.get(ANEEL_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(RAW_FILE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print("Download complete.")


def load_processed_data() -> pd.DataFrame:
    """Load, filter, and process ANEEL tariff data for BESS analysis.

    Pipeline:
    1. Load raw CSV (semicolon-separated, Latin-1 encoding)
    2. Parse numeric values (Brazilian comma-decimal format)
    3. Filter to Tarifa de Aplicação (actual applied tariffs)
    4. Filter to Grupo A subgroups (medium/high voltage)
    5. Filter to horossazonal modalities (Azul, Verde)
    6. Normalize seasonal tariff posts
    7. Keep only latest resolution per distributor
    """
    download_if_needed()

    df = pd.read_csv(RAW_FILE, sep=";", encoding="latin-1", dtype=str)

    # Parse numerics (Brazilian format: comma = decimal separator)
    for col in ("VlrTUSD", "VlrTE"):
        df[col] = df[col].str.replace(",", ".", regex=False).astype(float)

    # Parse dates
    df["DatInicioVigencia"] = pd.to_datetime(df["DatInicioVigencia"])

    # === Core Filters ===
    df = df[df["DscBaseTarifaria"] == "Tarifa de Aplicação"]
    df = df[df["DscSubGrupo"].isin(GRUPO_A)]
    df = df[df["DscModalidadeTarifaria"].isin(MODALITIES)]

    # Normalize tariff posts (merge seasonal variants)
    df["Posto"] = df["NomPostoTarifario"].map(POST_MAP)
    # Keep "Não se aplica" for Verde single demand rate
    mask_na = df["NomPostoTarifario"] == "Não se aplica"
    df.loc[mask_na, "Posto"] = "Única"
    df = df.dropna(subset=["Posto"])

    # Keep only latest resolution per distributor
    latest = df.groupby("SigAgente")["DatInicioVigencia"].transform("max")
    df = df[df["DatInicioVigencia"] == latest]

    # Computed columns
    df["VlrTotal"] = df["VlrTUSD"] + df["VlrTE"]
    df.rename(columns={"DscUnidadeTerciaria": "Unidade"}, inplace=True)

    return df


def get_energy_spreads(
    df: pd.DataFrame,
    subgrupo: str | None = None,
    modalidade: str | None = None,
) -> pd.DataFrame:
    """Calculate peak vs off-peak energy spread (R$/MWh) per distributor.

    Returns DataFrame with columns:
        SigAgente, DscModalidadeTarifaria, DscSubGrupo,
        TUSD_Ponta, TE_Ponta, Total_Ponta,
        TUSD_FP, TE_FP, Total_FP, SpreadEnergia
    """
    e = df[
        (df["Unidade"] == "MWh") & (df["Posto"].isin(["Ponta", "Fora ponta"]))
    ].copy()

    if subgrupo:
        e = e[e["DscSubGrupo"] == subgrupo]
    if modalidade:
        e = e[e["DscModalidadeTarifaria"] == modalidade]

    if e.empty:
        return pd.DataFrame()

    # Aggregate: mean tariff per (distributor, modality, subgroup, post)
    agg = (
        e.groupby(["SigAgente", "DscModalidadeTarifaria", "DscSubGrupo", "Posto"])
        .agg(TUSD=("VlrTUSD", "mean"), TE=("VlrTE", "mean"), Total=("VlrTotal", "mean"))
        .reset_index()
    )

    idx_cols = ["SigAgente", "DscModalidadeTarifaria", "DscSubGrupo"]

    ponta = (
        agg[agg["Posto"] == "Ponta"]
        .drop(columns="Posto")
        .set_index(idx_cols)
        .rename(columns={"TUSD": "TUSD_Ponta", "TE": "TE_Ponta", "Total": "Total_Ponta"})
    )
    fp = (
        agg[agg["Posto"] == "Fora ponta"]
        .drop(columns="Posto")
        .set_index(idx_cols)
        .rename(columns={"TUSD": "TUSD_FP", "TE": "TE_FP", "Total": "Total_FP"})
    )

    result = ponta.join(fp, how="inner").reset_index()
    result["SpreadEnergia"] = result["Total_Ponta"] - result["Total_FP"]

    return result


def get_demand_charges(
    df: pd.DataFrame, subgrupo: str | None = None
) -> pd.DataFrame:
    """Get peak/off-peak demand charges (R$/kW) for Azul modality.

    Returns DataFrame with: SigAgente, DscSubGrupo, Demanda_Ponta, Demanda_FP
    """
    d = df[
        (df["Unidade"] == "kW")
        & (df["DscModalidadeTarifaria"] == "Azul")
        & (df["Posto"].isin(["Ponta", "Fora ponta"]))
    ].copy()

    if subgrupo:
        d = d[d["DscSubGrupo"] == subgrupo]

    if d.empty:
        return pd.DataFrame()

    agg = d.groupby(["SigAgente", "DscSubGrupo", "Posto"])["VlrTUSD"].mean().reset_index()

    idx_cols = ["SigAgente", "DscSubGrupo"]

    ponta = (
        agg[agg["Posto"] == "Ponta"]
        .drop(columns="Posto")
        .set_index(idx_cols)
        .rename(columns={"VlrTUSD": "Demanda_Ponta"})
    )
    fp = (
        agg[agg["Posto"] == "Fora ponta"]
        .drop(columns="Posto")
        .set_index(idx_cols)
        .rename(columns={"VlrTUSD": "Demanda_FP"})
    )

    result = ponta.join(fp, how="inner").reset_index()
    return result
