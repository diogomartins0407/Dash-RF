"""Camada de extração de dados: Tesouro Direto, curva prefixada, séries BCB e ETFs de RF."""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import streamlit as st
import truststore
import yfinance as yf
from bcb import sgs
from dateutil.relativedelta import relativedelta

truststore.inject_into_ssl()

logger = logging.getLogger(__name__)

TESOURO_CSV_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)

TIPOS_PREFIXADOS = ["Tesouro Prefixado", "Tesouro Prefixado com Juros Semestrais"]
TIPOS_IPCA = ["Tesouro IPCA+", "Tesouro IPCA+ com Juros Semestrais"]
TIPO_SELIC = "Tesouro Selic"

CODIGOS_BCB = {"selic": 11, "cdi": 12, "ipca": 433}

TICKERS_ETF_RF = {
    "AUPO11": "AUPO11.SA",
    "LFTS11": "LFTS11.SA",
    "LFTI11": "LFTI11.SA",
    "NCDI11": "NCDI11.SA",
}


def get_data_referencia() -> date:
    """Data de referência fixa do projeto: hoje menos um mês."""
    return date.today() - relativedelta(months=1)


def _data_disponivel_mais_proxima(datas_disponiveis: pd.Series, data_alvo: date) -> pd.Timestamp:
    candidatas = datas_disponiveis[datas_disponiveis <= pd.Timestamp(data_alvo)]
    if candidatas.empty:
        raise ValueError(f"Nenhuma data disponível até {data_alvo.isoformat()}.")
    return candidatas.max()


@st.cache_data(ttl=86400, show_spinner="Baixando dados do Tesouro Direto...")
def _carregar_csv_tesouro_direto() -> pd.DataFrame:
    df = pd.read_csv(TESOURO_CSV_URL, sep=";", decimal=",", encoding="latin-1")
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(" ", "_")
    )
    df = df.rename(
        columns={
            "tipo_titulo": "tipo_titulo",
            "data_vencimento": "data_vencimento",
            "data_base": "data_base",
            "taxa_compra_manha": "taxa_compra",
            "taxa_venda_manha": "taxa_venda",
            "pu_compra_manha": "pu_compra",
            "pu_venda_manha": "pu_venda",
            "pu_base_manha": "pu_base",
        }
    )
    df["data_vencimento"] = pd.to_datetime(df["data_vencimento"], dayfirst=True)
    df["data_base"] = pd.to_datetime(df["data_base"], dayfirst=True)
    return df


def get_titulos_tesouro_direto(data_referencia: date | None = None) -> pd.DataFrame:
    """Retorna todos os títulos (PU/taxas) na data de referência (ou dia útil disponível anterior)."""
    data_referencia = data_referencia or get_data_referencia()
    df = _carregar_csv_tesouro_direto()
    data_base_usada = _data_disponivel_mais_proxima(df["data_base"], data_referencia)
    return df[df["data_base"] == data_base_usada].sort_values(
        ["tipo_titulo", "data_vencimento"]
    ).reset_index(drop=True)


def get_curva_prefixada(data_referencia: date | None = None) -> pd.DataFrame:
    """Curva de juros prefixada (proxy da curva DI) a partir das LTN/NTN-F por vencimento."""
    titulos = get_titulos_tesouro_direto(data_referencia)
    curva = titulos[titulos["tipo_titulo"].isin(TIPOS_PREFIXADOS)].copy()
    curva["prazo_anos"] = (
        curva["data_vencimento"] - curva["data_base"]
    ).dt.days / 365.25
    return curva[
        ["tipo_titulo", "data_vencimento", "prazo_anos", "taxa_compra", "taxa_venda", "pu_compra"]
    ].sort_values("data_vencimento").reset_index(drop=True)


def get_titulos_ipca(data_referencia: date | None = None) -> pd.DataFrame:
    """Títulos IPCA+ (NTN-B) na data de referência, usados na equação de Fisher."""
    titulos = get_titulos_tesouro_direto(data_referencia)
    ipca = titulos[titulos["tipo_titulo"].isin(TIPOS_IPCA)].copy()
    ipca["prazo_anos"] = (ipca["data_vencimento"] - ipca["data_base"]).dt.days / 365.25
    return ipca[
        ["tipo_titulo", "data_vencimento", "prazo_anos", "taxa_compra", "taxa_venda", "pu_compra"]
    ].sort_values("data_vencimento").reset_index(drop=True)


def get_titulos_selic(data_referencia: date | None = None) -> pd.DataFrame:
    """Títulos Tesouro Selic (LFT) na data de referência."""
    titulos = get_titulos_tesouro_direto(data_referencia)
    return titulos[titulos["tipo_titulo"] == TIPO_SELIC].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner="Baixando séries do Banco Central...")
def get_series_bcb(data_referencia: date | None = None, dias_historico: int = 60) -> pd.DataFrame:
    """Séries agregadas do BCB/SGS (Selic, CDI, IPCA) como complemento macro."""
    data_referencia = data_referencia or get_data_referencia()
    data_inicio = data_referencia - relativedelta(days=dias_historico)
    df = sgs.get(CODIGOS_BCB, start=data_inicio, end=data_referencia)
    return df


@st.cache_data(ttl=86400, show_spinner="Baixando cotações dos ETFs de Renda Fixa...")
def get_cotacoes_etfs(
    data_referencia: date | None = None, dias_historico: int = 180
) -> pd.DataFrame:
    """Histórico de fechamento (Close) dos ETFs de RF via yfinance, até a data de referência."""
    data_referencia = data_referencia or get_data_referencia()
    data_inicio = data_referencia - relativedelta(days=dias_historico)
    tickers = list(TICKERS_ETF_RF.values())
    dados = yf.download(
        tickers,
        start=data_inicio,
        end=data_referencia + relativedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if dados.empty:
        logger.warning("yfinance não retornou dados para os tickers %s", tickers)
        return pd.DataFrame()
    fechamento = dados["Close"] if "Close" in dados.columns.get_level_values(0) else dados
    fechamento.columns = [
        ticker.replace(".SA", "") for ticker in fechamento.columns
    ]
    return fechamento
