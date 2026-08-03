"""Aplicativo principal: casca visual + Feature 1 (Projeção de Valor Futuro) do Simulador de Renda Fixa."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai

from data_fetch import (
    get_cotacoes_etfs,
    get_curva_prefixada,
    get_data_referencia,
    get_series_bcb,
    get_titulos_ipca,
    get_titulos_selic,
)

load_dotenv()

st.set_page_config(
    page_title="Simulador de Renda Fixa",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, .stApp * {
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {
        font-family: 'Material Symbols Rounded' !important;
    }

    .stApp {
        background-color: #E2E8F0 !important;
    }

    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, label, span {
        color: #2D3748 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #DDE3EA;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.35rem 0;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGINAS = {
    "Valor Futuro": {
        "titulo": "Projeção de Valor Futuro",
        "descricao": (
            "Simule o valor futuro de um aporte inicial e aportes mensais, "
            "escolhendo entre Tesouro Direto e ETFs de Renda Fixa, já "
            "considerando a tributação de cada um."
        ),
    },
    "Marcação a Mercado": {
        "titulo": "Simulador de Marcação a Mercado",
        "descricao": (
            "Aqui o usuário vai simular o efeito de cenários de estresse na "
            "curva de juros sobre o preço de um título antes do vencimento, "
            "mostrando o ágio ou deságio resultante."
        ),
    },
    "Objetivos Financeiros": {
        "titulo": "Simulador de Objetivos Financeiros (Teoria dos Baldes)",
        "descricao": (
            "Aqui o usuário divide um único aporte mensal entre três objetivos "
            "simultâneos (curto, médio e longo prazo). Quando um objetivo é "
            "atingido, o dinheiro que ia para ele é redirecionado para acelerar "
            "os objetivos seguintes — o efeito cascata."
        ),
    },
    "Análise com IA": {
        "titulo": "Análise Generativa (LLM)",
        "descricao": (
            "Aqui o sistema vai exibir uma análise textual gerada por IA, "
            "traduzindo os resultados numéricos das outras abas em uma "
            "explicação simples e direta."
        ),
    },
    "Planejamento de Metas": {
        "titulo": "Planejamento de Metas",
        "descricao": (
            "Aqui o usuário vai informar um valor alvo futuro e o sistema vai "
            "calcular o aporte mensal necessário para atingir esse objetivo no "
            "prazo estipulado."
        ),
    },
}

ATIVOS_VALOR_FUTURO = {
    "Tesouro Prefixado": {"categoria": "tesouro", "fonte": "prefixado"},
    "Tesouro IPCA+": {"categoria": "tesouro", "fonte": "ipca"},
    "Tesouro Selic": {"categoria": "tesouro", "fonte": "selic"},
    "AUPO11": {"categoria": "etf", "ticker": "AUPO11"},
    "LFTS11": {"categoria": "etf", "ticker": "LFTS11"},
    "LFTI11": {"categoria": "etf", "ticker": "LFTI11"},
    "NCDI11": {"categoria": "etf", "ticker": "NCDI11"},
}

TIPOS_TITULO_MARCACAO = {
    "Tesouro Prefixado": "prefixado",
    "Tesouro IPCA+": "ipca",
    "Tesouro Selic": "selic",
}

DELTA_CENARIO_OTIMISTA_PP = -1.5
DELTA_CENARIO_ESTRESSE_PP = 1.5
DELTA_TAXA_MAXIMO_PP = 3.0

TAXA_PADRAO_FALLBACK = 0.10
TAXA_INFLACAO_APORTES = 0.045
TAXA_REINVESTIMENTO_PADRAO_PCT = 8.5
TAXA_POUPANCA_MENSAL_REGRA_ALTA = 0.005
LIMIAR_SELIC_POUPANCA_PCT = 8.5
PERCENTUAL_POUPANCA_SOBRE_SELIC = 0.7
DIAS_UTEIS_ANO = 252
PRAZO_MAXIMO_ANOS = 30

ORDEM_BALDES = ["curto", "medio", "longo"]
FATIA_INICIAL_BALDES = {"curto": 0.40, "medio": 0.30, "longo": 0.30}
LIMITE_MESES_SEGURANCA_BALDES = 1200

TIPO_EVENTO_APORTE_EXTRA = "Aporte Extra Único"
TIPO_EVENTO_NOVO_APORTE_MENSAL = "Novo Aporte Mensal"
TIPOS_EVENTO = [TIPO_EVENTO_APORTE_EXTRA, TIPO_EVENTO_NOVO_APORTE_MENSAL]

COLUNAS_TABELA_EVENTOS = ["Ano", "Tipo de Evento", "Valor (R$)"]

CHAVE_APORTE_INICIAL = "valor_futuro_aporte_inicial"
CHAVE_APORTE_MENSAL = "valor_futuro_aporte_mensal"
CHAVE_PRAZO_ANOS = "valor_futuro_prazo_anos"
CHAVE_ATIVO = "valor_futuro_ativo"
CHAVE_REAJUSTE_INFLACAO = "valor_futuro_reajuste_inflacao"
CHAVE_TAXA_REINVESTIMENTO = "valor_futuro_taxa_reinvestimento"
CHAVE_TABELA_EVENTOS_EDITOR = "valor_futuro_tabela_eventos_editor"
CHAVE_VALORES_REAIS = "valor_futuro_valores_reais"
CHAVE_CONTADOR_RESET = "valor_futuro_reset_contador"

CHAVE_MM_TIPO_TITULO = "marcacao_tipo_titulo"
CHAVE_MM_VENCIMENTO = "marcacao_vencimento"
CHAVE_MM_DELTA_TAXA = "marcacao_delta_taxa"

CHAVE_OF_APORTE_MENSAL_TOTAL = "objetivos_aporte_mensal_total"
CHAVE_OF_TAXA_ESPERADA = "objetivos_taxa_esperada"
CHAVE_OF_NOME_CURTO = "objetivos_nome_curto"
CHAVE_OF_VALOR_CURTO = "objetivos_valor_curto"
CHAVE_OF_NOME_MEDIO = "objetivos_nome_medio"
CHAVE_OF_VALOR_MEDIO = "objetivos_valor_medio"
CHAVE_OF_NOME_LONGO = "objetivos_nome_longo"
CHAVE_OF_VALOR_LONGO = "objetivos_valor_longo"

MODELO_GEMINI = "gemini-flash-latest"

TABELA_IOF_REGRESSIVO = {
    1: 0.96, 2: 0.93, 3: 0.90, 4: 0.86, 5: 0.83, 6: 0.80, 7: 0.76, 8: 0.73,
    9: 0.70, 10: 0.66, 11: 0.63, 12: 0.60, 13: 0.56, 14: 0.53, 15: 0.50,
    16: 0.46, 17: 0.43, 18: 0.40, 19: 0.36, 20: 0.33, 21: 0.30, 22: 0.26,
    23: 0.23, 24: 0.20, 25: 0.16, 26: 0.13, 27: 0.10, 28: 0.06, 29: 0.03,
    30: 0.0,
}


def _taxa_titulo_mais_proximo(curva: pd.DataFrame, prazo_anos: float) -> float:
    if curva.empty:
        raise ValueError("Curva de títulos sem dados disponíveis.")
    idx = (curva["prazo_anos"] - prazo_anos).abs().idxmin()
    return float(curva.loc[idx, "taxa_compra"])


def _retorno_anualizado_etf(precos: pd.Series) -> float:
    precos = precos.dropna()
    if len(precos) < 2:
        raise ValueError("Histórico de cotações insuficiente para o ETF.")
    dias = (precos.index[-1] - precos.index[0]).days
    if dias <= 0:
        raise ValueError("Histórico de cotações insuficiente para o ETF.")
    retorno_total = precos.iloc[-1] / precos.iloc[0] - 1
    return (1 + retorno_total) ** (365 / dias) - 1


def obter_selic_cdi_atuais() -> tuple[float, float]:
    """Retorna (selic_pct, cdi_pct) anuais vigentes a partir da série mais recente do BCB.

    As séries 11 e 12 do SGS trazem a taxa diária (% ao dia); anualiza-se pelo
    padrão de mercado de 252 dias úteis.
    """
    series = get_series_bcb()
    selic_diaria_pct = float(series["selic"].dropna().iloc[-1])
    cdi_diaria_pct = float(series["cdi"].dropna().iloc[-1])
    selic_pct = ((1 + selic_diaria_pct / 100) ** DIAS_UTEIS_ANO - 1) * 100
    cdi_pct = ((1 + cdi_diaria_pct / 100) ** DIAS_UTEIS_ANO - 1) * 100
    return selic_pct, cdi_pct


def calcular_taxa_poupanca_mensal(selic_anual_pct: float) -> float:
    """Regra oficial da poupança pós-2012: 0,5% a.m. se a Selic estiver acima de 8,5% a.a.,
    senão 70% da Selic a.a. convertidos a taxa mensal. A TR é assumida como zero (valor
    residual nos últimos anos), simplificação evidenciada na interface.
    """
    if selic_anual_pct > LIMIAR_SELIC_POUPANCA_PCT:
        return TAXA_POUPANCA_MENSAL_REGRA_ALTA
    return (1 + PERCENTUAL_POUPANCA_SOBRE_SELIC * (selic_anual_pct / 100)) ** (1 / 12) - 1


def obter_inflacao_implicita(prazo_anos: float) -> tuple[float, str]:
    """Inflação implícita de mercado via equação de Fisher, comparando o Tesouro Prefixado
    com o Tesouro IPCA+ de prazo mais próximo ao simulado (Seção 5 do CLAUDE.md).
    """
    taxa_pre_pct = _taxa_titulo_mais_proximo(get_curva_prefixada(), prazo_anos)
    taxa_real_pct = _taxa_titulo_mais_proximo(get_titulos_ipca(), prazo_anos)
    inflacao_implicita = (1 + taxa_pre_pct / 100) / (1 + taxa_real_pct / 100) - 1
    metodologia = (
        "Inflação Implícita = (1 + Taxa Prefixada) / (1 + Taxa Real IPCA+) - 1 = "
        f"(1 + {taxa_pre_pct:.2f}%) / (1 + {taxa_real_pct:.2f}%) - 1 = "
        f"{inflacao_implicita * 100:.2f}% a.a., com base nos títulos de prazo mais "
        "próximo ao prazo simulado."
    )
    return inflacao_implicita, metodologia


def converter_para_valores_reais(df: pd.DataFrame, inflacao_anual: float) -> pd.DataFrame:
    """Deflaciona as colunas monetárias mês a mês, trazendo tudo a poder de compra de hoje."""
    colunas_monetarias = [
        "saldo_inicial", "saldo_novos_aportes", "saldo_poupanca",
        "capital_investido", "saldo_bruto", "juros_acumulados",
    ]
    fator = (1 + inflacao_anual) ** (df["mes"] / 12)
    df_real = df.copy()
    df_real[colunas_monetarias] = df_real[colunas_monetarias].div(fator, axis=0)
    return df_real


def obter_taxa_anual_bruta(ativo_nome: str, prazo_anos: float) -> tuple[float, str]:
    """Busca a taxa anual bruta do ativo escolhido a partir do data_fetch.py."""
    info = ATIVOS_VALOR_FUTURO[ativo_nome]

    if info["categoria"] == "tesouro":
        if info["fonte"] == "prefixado":
            taxa_pct = _taxa_titulo_mais_proximo(get_curva_prefixada(), prazo_anos)
            return taxa_pct / 100, "Taxa do Tesouro Prefixado (LTN/NTN-F) mais próxima do prazo escolhido."

        if info["fonte"] == "ipca":
            taxa_pct = _taxa_titulo_mais_proximo(get_titulos_ipca(), prazo_anos)
            return taxa_pct / 100, "Taxa real do Tesouro IPCA+ (NTN-B) mais próxima do prazo escolhido."

        titulos_selic = get_titulos_selic()
        if titulos_selic.empty:
            raise ValueError("Sem títulos Tesouro Selic disponíveis.")
        spread_pct = float(titulos_selic["taxa_compra"].iloc[0])
        selic_pct, _ = obter_selic_cdi_atuais()
        return (selic_pct + spread_pct) / 100, "Selic vigente ajustada pelo spread de compra do Tesouro Selic."

    precos = get_cotacoes_etfs()[info["ticker"]]
    taxa = _retorno_anualizado_etf(precos)
    return taxa, "Retorno anualizado com base no histórico recente de cotações do ETF."


def obter_titulos_zero_cupom(categoria: str) -> pd.DataFrame:
    """Retorna só os títulos sem cupom (LTN / NTN-B Principal), que permitem
    reprecificação exata de PU sem modelar fluxo de cupons semestrais.
    """
    if categoria == "prefixado":
        curva = get_curva_prefixada()
        return curva[curva["tipo_titulo"] == "Tesouro Prefixado"].reset_index(drop=True)
    curva = get_titulos_ipca()
    return curva[curva["tipo_titulo"] == "Tesouro IPCA+"].reset_index(drop=True)


def calcular_du_uteis(data_vencimento: pd.Timestamp) -> int:
    """Dias úteis entre a data de referência dos dados e o vencimento (fins de semana
    apenas; feriados não são considerados, simplificação evidenciada na interface).
    """
    return int(np.busday_count(get_data_referencia(), data_vencimento.date()))


def calcular_duration_anos(du_uteis: int) -> float:
    """Duration de Macaulay de um título sem cupom: coincide com o prazo até o vencimento."""
    return du_uteis / DIAS_UTEIS_ANO


def simular_pu_estressado(
    pu_atual: float, taxa_atual_pct: float, delta_taxa_pp: float, du_uteis: int
) -> tuple[float, float]:
    """Reprecifica um título sem cupom sob uma nova taxa, a partir do PU e da taxa atuais.

    Exato para título sem cupom: PU = VN_ou_VNA / (1+taxa)^(du/252), então
    PU_novo / PU_atual = [(1+taxa_atual)/(1+taxa_nova)]^(du/252), sem precisar
    conhecer VN/VNA explicitamente.
    """
    taxa_atual = taxa_atual_pct / 100
    taxa_nova_pct = taxa_atual_pct + delta_taxa_pp
    taxa_nova = taxa_nova_pct / 100
    pu_novo = pu_atual * ((1 + taxa_atual) / (1 + taxa_nova)) ** (du_uteis / DIAS_UTEIS_ANO)
    return pu_novo, taxa_nova_pct


def montar_tabela_sensibilidade(titulos: pd.DataFrame, delta_taxa_pp: float) -> pd.DataFrame:
    """Aplica o mesmo choque de taxa (paralelo) a todos os vencimentos de um tipo de título,
    para comparar qual deles reage mais ao cenário — quanto maior a duration, maior o efeito.
    """
    linhas = []
    for linha in titulos.itertuples():
        du_uteis = calcular_du_uteis(linha.data_vencimento)
        duration_anos = calcular_duration_anos(du_uteis)
        pu_atual = float(linha.pu_compra)
        taxa_atual_pct = float(linha.taxa_compra)
        pu_novo, taxa_nova_pct = simular_pu_estressado(pu_atual, taxa_atual_pct, delta_taxa_pp, du_uteis)
        linhas.append(
            {
                "Vencimento": linha.data_vencimento.strftime("%d/%m/%Y"),
                "Duration (anos)": duration_anos,
                "Taxa Atual (% a.a.)": taxa_atual_pct,
                "PU Atual (R$)": pu_atual,
                "Taxa Simulada (% a.a.)": taxa_nova_pct,
                "PU Simulado (R$)": pu_novo,
                "Ágio/Deságio (%)": (pu_novo / pu_atual - 1) * 100,
            }
        )
    return pd.DataFrame(linhas).sort_values("Ágio/Deságio (%)", ascending=False).reset_index(drop=True)


def montar_prompt_analise_marcacao(resultados: dict) -> str:
    return (
        "Você é um analista financeiro. Explique o resultado de uma simulação de "
        "marcação a mercado de um título de renda fixa para um investidor leigo, "
        "em no máximo 150 palavras, em português do Brasil. O investidor está "
        f"avaliando vender o título antes do vencimento ({resultados['prazo_restante_anos']:.1f} "
        f"anos restantes, duration de {resultados['duration_anos']:.1f} anos) sob um cenário em que "
        f"a taxa de juros do título {'sobe' if resultados['delta_taxa_pp'] > 0 else 'cai'} "
        f"{abs(resultados['delta_taxa_pp']):.2f} pontos percentuais, de "
        f"{resultados['taxa_atual_pct']:.2f}% para {resultados['taxa_nova_pct']:.2f}% a.a. "
        "Explique de forma simples por que o preço se move na direção oposta à taxa "
        "(quanto maior a duration, maior o efeito), e deixe claro que esse "
        "ágio/deságio só afeta quem vende antes do vencimento — quem carrega o "
        "título até o fim recebe a taxa contratada, independentemente do que "
        "aconteceu no meio do caminho. Não utilize emojis em nenhuma parte da "
        "resposta.\n\n"
        f"PU Atual: {formatar_moeda(resultados['pu_atual'])}\n"
        f"PU Simulado: {formatar_moeda(resultados['pu_novo'])}\n"
        f"Ágio/Deságio: {formatar_moeda(resultados['pu_novo'] - resultados['pu_atual'])} "
        f"({resultados['variacao_pct']:+.2f}%)\n"
    )


def simular_baldes_cascata(
    aporte_mensal_total: float,
    taxa_anual: float,
    valores_alvo: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    """Distribui um aporte mensal único entre três baldes (curto/médio/longo prazo).

    Quando um balde atinge sua meta, seu saldo é travado (para de receber aporte e de
    render) e a fatia do aporte que ia para ele é redirecionada ao próximo balde ainda
    aberto na ordem curto -> médio -> longo (efeito cascata). Roda mês a mês até que os
    três baldes atinjam a meta, com um limite de segurança de meses.
    """
    taxa_mensal = (1 + taxa_anual) ** (1 / 12) - 1
    fatia = dict(FATIA_INICIAL_BALDES)
    saldo = {chave: 0.0 for chave in ORDEM_BALDES}
    trancado = {chave: False for chave in ORDEM_BALDES}
    mes_atingido: dict[str, int | None] = {chave: None for chave in ORDEM_BALDES}

    registros = [{"mes": 0, **{f"saldo_{chave}": 0.0 for chave in ORDEM_BALDES}}]

    mes = 0
    while not all(trancado.values()) and mes < LIMITE_MESES_SEGURANCA_BALDES:
        mes += 1
        for chave in ORDEM_BALDES:
            if trancado[chave]:
                continue
            saldo[chave] *= 1 + taxa_mensal
            saldo[chave] += fatia[chave] * aporte_mensal_total
            if saldo[chave] >= valores_alvo[chave]:
                trancado[chave] = True
                mes_atingido[chave] = mes
                fatia_liberada = fatia[chave]
                fatia[chave] = 0.0
                indice = ORDEM_BALDES.index(chave)
                destino = next((c for c in ORDEM_BALDES[indice + 1 :] if not trancado[c]), None)
                if destino is None:
                    destino = next((c for c in ORDEM_BALDES if not trancado[c]), None)
                if destino is not None:
                    fatia[destino] += fatia_liberada
        registros.append({"mes": mes, **{f"saldo_{chave}": saldo[chave] for chave in ORDEM_BALDES}})

    return pd.DataFrame(registros), mes_atingido


def formatar_anos_meses(total_meses: int | None) -> str:
    if total_meses is None:
        return f"Não atingida em {LIMITE_MESES_SEGURANCA_BALDES // 12} anos"
    anos, meses = divmod(total_meses, 12)
    partes = []
    if anos:
        partes.append(f"{anos} ano{'s' if anos != 1 else ''}")
    if meses or not partes:
        partes.append(f"{meses} {'mês' if meses == 1 else 'meses'}")
    return " e ".join(partes)


def montar_prompt_analise_baldes(resultados: dict) -> str:
    return (
        "Você é um analista financeiro. Explique para um investidor leigo, em português "
        "do Brasil e em no máximo 150 palavras, o resultado de uma simulação de "
        "planejamento financeiro com três objetivos simultâneos (curto, médio e longo "
        "prazo) usando a técnica dos baldes: um único aporte mensal é dividido entre as "
        "três metas e, assim que uma meta é atingida, o valor que ia para ela é "
        "redirecionado para acelerar as metas seguintes (efeito cascata). Destaque a "
        "força da consistência e do efeito cascata na alocação de capital, usando os "
        "prazos abaixo como exemplo concreto. Não utilize emojis em nenhuma parte da "
        "resposta.\n\n"
        f"Aporte mensal total: {formatar_moeda(resultados['aporte_mensal_total'])}\n"
        f"Taxa de rendimento esperada: {resultados['taxa_esperada_pct']:.2f}% a.a.\n"
        f"{resultados['nome_curto']} (meta de {formatar_moeda(resultados['valor_curto'])}): "
        f"atingida em {resultados['tempo_curto']}\n"
        f"{resultados['nome_medio']} (meta de {formatar_moeda(resultados['valor_medio'])}): "
        f"atingida em {resultados['tempo_medio']}\n"
        f"{resultados['nome_longo']} (meta de {formatar_moeda(resultados['valor_longo'])}): "
        f"atingida em {resultados['tempo_longo']}\n"
    )


def _preparar_eventos(eventos_df: pd.DataFrame | None, prazo_anos: int) -> pd.DataFrame:
    """Valida, limpa e ordena cronologicamente a tabela de eventos vinda do data_editor."""
    colunas_vazias = {coluna: pd.Series(dtype="object") for coluna in COLUNAS_TABELA_EVENTOS}
    if eventos_df is None or eventos_df.empty:
        return pd.DataFrame(colunas_vazias)

    eventos_validos = eventos_df.dropna(subset=COLUNAS_TABELA_EVENTOS).copy()
    if eventos_validos.empty:
        return pd.DataFrame(colunas_vazias)

    eventos_validos["Ano"] = eventos_validos["Ano"].astype(int).clip(lower=1, upper=prazo_anos)
    eventos_validos["Valor (R$)"] = eventos_validos["Valor (R$)"].astype(float)
    eventos_validos = eventos_validos[eventos_validos["Tipo de Evento"].isin(TIPOS_EVENTO)]
    return eventos_validos.sort_values("Ano").reset_index(drop=True)


def simular_valor_futuro(
    aporte_inicial: float,
    aporte_mensal: float,
    taxa_anual: float,
    taxa_reinvestimento_anual: float,
    prazo_anos: int,
    taxa_poupanca_mensal: float,
    eventos_df: pd.DataFrame | None = None,
    reajustar_inflacao: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Evolução mês a mês do capital investido e do saldo bruto, a juros compostos.

    Loop iterativo (não fórmula fechada) que separa dois baldes de saldo para
    isolar o risco de reinvestimento: o aporte inicial rende à taxa do ativo
    escolhido (`taxa_anual`), enquanto todo dinheiro novo (aporte mensal
    recorrente, aportes extras e alterações de aporte da tabela de eventos)
    rende à taxa de reinvestimento informada pelo usuário. Também suporta o
    reajuste anual dos aportes pela inflação. Em paralelo, calcula uma terceira
    linha de referência (`saldo_poupanca`) recebendo os mesmos aportes a uma
    taxa fixa conservadora, apenas para contraste visual no gráfico.
    """
    meses = prazo_anos * 12
    taxa_mensal_inicial = (1 + taxa_anual) ** (1 / 12) - 1
    taxa_mensal_reinvestimento = (1 + taxa_reinvestimento_anual) ** (1 / 12) - 1
    eventos_ordenados = _preparar_eventos(eventos_df, prazo_anos)

    saldo_inicial = aporte_inicial
    saldo_novos_aportes = 0.0
    saldo_poupanca = aporte_inicial
    capital_investido = aporte_inicial
    aporte_mensal_atual = aporte_mensal

    eventos_grafico: list[dict] = []
    registros = [
        {
            "mes": 0,
            "saldo_inicial": saldo_inicial,
            "saldo_novos_aportes": saldo_novos_aportes,
            "saldo_poupanca": saldo_poupanca,
            "capital_investido": capital_investido,
            "saldo_bruto": saldo_inicial + saldo_novos_aportes,
        }
    ]

    for mes in range(1, meses + 1):
        saldo_inicial *= 1 + taxa_mensal_inicial
        saldo_novos_aportes *= 1 + taxa_mensal_reinvestimento
        saldo_poupanca *= 1 + taxa_poupanca_mensal

        if mes % 12 == 1:
            ano_atual = (mes - 1) // 12 + 1
            eventos_do_ano = eventos_ordenados[eventos_ordenados["Ano"] == ano_atual]
            for _, evento in eventos_do_ano.iterrows():
                valor_evento = float(evento["Valor (R$)"])
                if evento["Tipo de Evento"] == TIPO_EVENTO_APORTE_EXTRA:
                    saldo_novos_aportes += valor_evento
                    saldo_poupanca += valor_evento
                    capital_investido += valor_evento
                    eventos_grafico.append({"mes": mes, "texto": "Aporte Extra"})
                elif evento["Tipo de Evento"] == TIPO_EVENTO_NOVO_APORTE_MENSAL:
                    aporte_mensal_atual = valor_evento
                    eventos_grafico.append({"mes": mes, "texto": "Aumento Aporte"})

        if reajustar_inflacao and mes % 12 == 0:
            aporte_mensal_atual *= 1 + TAXA_INFLACAO_APORTES

        saldo_novos_aportes += aporte_mensal_atual
        saldo_poupanca += aporte_mensal_atual
        capital_investido += aporte_mensal_atual

        registros.append(
            {
                "mes": mes,
                "saldo_inicial": saldo_inicial,
                "saldo_novos_aportes": saldo_novos_aportes,
                "saldo_poupanca": saldo_poupanca,
                "capital_investido": capital_investido,
                "saldo_bruto": saldo_inicial + saldo_novos_aportes,
            }
        )

    df = pd.DataFrame(registros)
    df["juros_acumulados"] = df["saldo_bruto"] - df["capital_investido"]
    return df, eventos_grafico


def aliquota_iof(dias_corridos: int) -> float:
    if dias_corridos >= 30:
        return 0.0
    return TABELA_IOF_REGRESSIVO.get(max(dias_corridos, 1), 0.0)


def aliquota_ir_regressiva(dias_corridos: int) -> float:
    if dias_corridos <= 180:
        return 0.225
    if dias_corridos <= 360:
        return 0.20
    if dias_corridos <= 720:
        return 0.175
    return 0.15


def calcular_impostos(categoria: str, lucro_bruto: float, dias_corridos: int) -> dict:
    lucro_bruto = max(lucro_bruto, 0.0)
    if categoria == "tesouro":
        iof = lucro_bruto * aliquota_iof(dias_corridos)
        ir = (lucro_bruto - iof) * aliquota_ir_regressiva(dias_corridos)
        return {"iof": iof, "ir": ir, "total": iof + ir}
    ir = lucro_bruto * 0.15
    return {"iof": 0.0, "ir": ir, "total": ir}


def formatar_moeda(valor: float, decimais: int = 2) -> str:
    texto = f"R$ {valor:,.{decimais}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def _tabela_eventos_vazia() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ano": pd.Series(dtype="int"),
            "Tipo de Evento": pd.Series(dtype="str"),
            "Valor (R$)": pd.Series(dtype="float"),
        }
    )


def montar_prompt_analise(resultados: dict) -> str:
    return (
        "Você é um analista financeiro. Explique o resultado de simulação de "
        "investimento em renda fixa abaixo para um investidor leigo, em no "
        "máximo 150 palavras, em português do Brasil, com foco no efeito dos "
        "juros compostos ao longo do prazo simulado. O cenário considera que "
        "os novos aportes foram reinvestidos a uma taxa de "
        f"{resultados['taxa_reinvestimento_pct']:.2f}% a.a., diferente da taxa "
        f"inicial de {resultados['taxa_ativo_pct']:.2f}% a.a. Mencione "
        "brevemente o impacto dessa diferença de taxa na acumulação. Inicie ou "
        "finalize a sua análise destacando que "
        f"{resultados['esforco_investidor_pct']:.0f}% do patrimônio final veio "
        "do esforço do investidor (aportes) e "
        f"{resultados['esforco_tempo_pct']:.0f}% veio do tempo (juros "
        "compostos). Use essa proporção para tangibilizar o poder dos juros. "
        "Não utilize emojis em nenhuma parte da resposta.\n\n"
        f"Total Investido: {formatar_moeda(resultados['total_investido'])}\n"
        f"Saldo Bruto: {formatar_moeda(resultados['saldo_bruto'])}\n"
        f"Impostos Pagos: {formatar_moeda(resultados['impostos'])}\n"
        f"Saldo Líquido: {formatar_moeda(resultados['saldo_liquido'])}\n"
    )


def gerar_analise_ia(prompt: str) -> str:
    """Chama o Gemini para gerar uma análise textual curta a partir de um prompt já pronto."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não configurada no .env.")

    cliente = genai.Client(api_key=api_key)
    resposta = cliente.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
    )
    return resposta.text


def renderizar_grafico_area_empilhada(df: pd.DataFrame, eventos: list[dict] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["mes"],
            y=df["capital_investido"],
            name="Capital Investido",
            mode="lines",
            stackgroup="one",
            line=dict(width=0.5, color="#173451"),
            fillcolor="#173451",
            hovertemplate="Mês %{x}<br>Capital Investido: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["mes"],
            y=df["juros_acumulados"],
            name="Juros Acumulados",
            mode="lines",
            stackgroup="one",
            line=dict(width=0.5, color="#F19828"),
            fillcolor="#F19828",
            hovertemplate="Mês %{x}<br>Juros Acumulados: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["mes"],
            y=df["saldo_poupanca"],
            name="Poupança (Referência)",
            mode="lines",
            line=dict(width=1.5, color="#CBD5E1", dash="dash"),
            hovertemplate="Mês %{x}<br>Poupança (Referência): R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#2D3748"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        xaxis=dict(title="Meses", showgrid=False, zeroline=False),
        yaxis=dict(title="Valor (R$)", showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickprefix="R$ "),
    )

    posicoes_anotacao = ["top left", "top right"]
    for indice, evento in enumerate(eventos or []):
        fig.add_vline(
            x=evento["mes"],
            line_dash="dash",
            line_color="#8AA0B9",
            annotation_text=evento["texto"],
            annotation_position=posicoes_anotacao[indice % len(posicoes_anotacao)],
            annotation_font_color="#2D3748",
            annotation_font_size=11,
        )

    return fig


def renderizar_valor_futuro() -> None:
    st.header(PAGINAS["Valor Futuro"]["titulo"])
    st.write(PAGINAS["Valor Futuro"]["descricao"])

    sufixo_reset = st.session_state.get(CHAVE_CONTADOR_RESET, 0)

    try:
        selic_pct_atual, cdi_pct_atual = obter_selic_cdi_atuais()
    except Exception:
        selic_pct_atual, cdi_pct_atual = None, None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        aporte_inicial = st.number_input(
            "Aporte Inicial (R$)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key=f"{CHAVE_APORTE_INICIAL}_{sufixo_reset}",
        )
    with col2:
        aporte_mensal = st.number_input(
            "Aporte Mensal (R$)",
            min_value=0.0,
            value=200.0,
            step=50.0,
            key=f"{CHAVE_APORTE_MENSAL}_{sufixo_reset}",
        )
    with col3:
        prazo_anos = st.slider(
            "Prazo de Investimento (anos)",
            min_value=1,
            max_value=PRAZO_MAXIMO_ANOS,
            value=5,
            key=f"{CHAVE_PRAZO_ANOS}_{sufixo_reset}",
        )
    with col4:
        ativo_selecionado = st.selectbox(
            "Ativo", list(ATIVOS_VALOR_FUTURO.keys()), key=f"{CHAVE_ATIVO}_{sufixo_reset}"
        )

    with st.expander("Configurações Avançadas de Aportes"):
        taxa_reinvestimento_padrao = cdi_pct_atual if cdi_pct_atual is not None else TAXA_REINVESTIMENTO_PADRAO_PCT
        taxa_reinvestimento_pct = st.number_input(
            "Taxa de Reinvestimento para Novos Aportes (% a.a.)",
            min_value=0.0,
            value=taxa_reinvestimento_padrao,
            step=0.1,
            key=f"{CHAVE_TAXA_REINVESTIMENTO}_{sufixo_reset}",
        )
        if cdi_pct_atual is not None:
            st.caption(
                f"Pré-preenchido com o CDI vigente ({cdi_pct_atual:.2f}% a.a.); edite se "
                "quiser simular outra premissa de reinvestimento."
            )
        else:
            st.caption(
                f"Não foi possível obter o CDI vigente agora; pré-preenchido com "
                f"{TAXA_REINVESTIMENTO_PADRAO_PCT:.2f}% a.a. como premissa padrão."
            )
        st.caption(
            "O aporte inicial continua rendendo à taxa do ativo escolhido acima; "
            "todo dinheiro novo (aporte mensal e os eventos cadastrados abaixo) "
            "rende a esta taxa de reinvestimento, isolando o risco de reinvestir "
            "novos aportes a uma taxa diferente da atual."
        )

        st.caption(
            "Cadastre quantos eventos quiser: aportes extraordinários pontuais "
            "ou alterações do aporte mensal recorrente, cada um no ano em que "
            "deve entrar em vigor."
        )

        eventos_editados = st.data_editor(
            _tabela_eventos_vazia(),
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"{CHAVE_TABELA_EVENTOS_EDITOR}_{sufixo_reset}",
            column_config={
                "Ano": st.column_config.NumberColumn(
                    "Ano", min_value=1, max_value=PRAZO_MAXIMO_ANOS, step=1, format="%d", width="small"
                ),
                "Tipo de Evento": st.column_config.SelectboxColumn(
                    "Tipo de Evento", options=TIPOS_EVENTO
                ),
                "Valor (R$)": st.column_config.NumberColumn(
                    "Valor (R$)", min_value=0.0, step=100.0, format="R$ %.2f"
                ),
            },
        )

        reajustar_inflacao = st.checkbox(
            "Reajustar aportes anualmente pela inflação",
            key=f"{CHAVE_REAJUSTE_INFLACAO}_{sufixo_reset}",
        )
        if reajustar_inflacao:
            st.caption(f"Premissa fixa de IPCA de {TAXA_INFLACAO_APORTES * 100:.1f}% a.a. sobre os aportes.")

    try:
        taxa_anual, observacao = obter_taxa_anual_bruta(ativo_selecionado, prazo_anos)
    except Exception as exc:
        taxa_anual = TAXA_PADRAO_FALLBACK
        observacao = (
            f"Não foi possível obter a taxa de mercado agora ({exc}); "
            f"usando taxa padrão de {TAXA_PADRAO_FALLBACK * 100:.0f}% a.a. para a simulação."
        )
        st.warning(observacao)

    taxa_reinvestimento_anual = taxa_reinvestimento_pct / 100

    if selic_pct_atual is not None:
        taxa_poupanca_mensal = calcular_taxa_poupanca_mensal(selic_pct_atual)
        regra_poupanca = (
            "0,50% a.m." if selic_pct_atual > LIMIAR_SELIC_POUPANCA_PCT
            else f"{PERCENTUAL_POUPANCA_SOBRE_SELIC * 100:.0f}% da Selic a.a."
        )
        observacao_poupanca = (
            f"Linha de referência 'Poupança' segue a regra oficial pós-2012 com a Selic "
            f"vigente de {selic_pct_atual:.2f}% a.a. ({regra_poupanca} + TR, aqui simplificada com TR = 0)."
        )
    else:
        taxa_poupanca_mensal = TAXA_POUPANCA_MENSAL_REGRA_ALTA
        observacao_poupanca = (
            "Não foi possível obter a Selic vigente agora; linha de referência 'Poupança' "
            f"usando {TAXA_POUPANCA_MENSAL_REGRA_ALTA * 100:.2f}% a.m. como premissa padrão."
        )

    df, eventos = simular_valor_futuro(
        aporte_inicial,
        aporte_mensal,
        taxa_anual,
        taxa_reinvestimento_anual,
        prazo_anos,
        taxa_poupanca_mensal,
        eventos_df=eventos_editados,
        reajustar_inflacao=reajustar_inflacao,
    )
    linha_final = df.iloc[-1]

    categoria = ATIVOS_VALOR_FUTURO[ativo_selecionado]["categoria"]
    dias_corridos = prazo_anos * 365

    aportes_futuros_total = linha_final["capital_investido"] - aporte_inicial
    lucro_balde_inicial = linha_final["saldo_inicial"] - aporte_inicial
    lucro_balde_novo = linha_final["saldo_novos_aportes"] - aportes_futuros_total
    lucro_tributavel = lucro_balde_inicial + lucro_balde_novo

    impostos = calcular_impostos(categoria, lucro_tributavel, dias_corridos)
    saldo_liquido = linha_final["saldo_bruto"] - impostos["total"]

    if linha_final["saldo_bruto"] > 0:
        esforco_investidor_pct = linha_final["capital_investido"] / linha_final["saldo_bruto"] * 100
        esforco_tempo_pct = linha_final["juros_acumulados"] / linha_final["saldo_bruto"] * 100
    else:
        esforco_investidor_pct = 0.0
        esforco_tempo_pct = 0.0

    st.caption(
        f"Taxa bruta anual utilizada: {taxa_anual * 100:.2f}% ao ano. {observacao} "
        f"Novos aportes reinvestidos a {taxa_reinvestimento_pct:.2f}% ao ano. {observacao_poupanca}"
    )

    try:
        inflacao_implicita, metodologia_inflacao = obter_inflacao_implicita(prazo_anos)
    except Exception:
        inflacao_implicita, metodologia_inflacao = None, None

    exibir_valores_reais = st.checkbox(
        "Exibir valores em poder de compra de hoje (descontar a inflação implícita de mercado)",
        key=f"{CHAVE_VALORES_REAIS}_{sufixo_reset}",
        disabled=inflacao_implicita is None,
    )
    with st.expander("Como calculamos a inflação implícita de mercado?"):
        if metodologia_inflacao:
            st.write(metodologia_inflacao)
        else:
            st.write(
                "Não foi possível calcular a inflação implícita agora, pois faltam "
                "títulos Prefixados ou IPCA+ com prazo próximo ao simulado."
            )

    if exibir_valores_reais and inflacao_implicita is not None:
        df_exibicao = converter_para_valores_reais(df, inflacao_implicita)
        fator_deflator_final = (1 + inflacao_implicita) ** prazo_anos
        st.caption(
            f"Valores abaixo em poder de compra de hoje, descontados a "
            f"{inflacao_implicita * 100:.2f}% a.a. de inflação implícita."
        )
    else:
        df_exibicao = df
        fator_deflator_final = 1.0
        st.caption("Valores abaixo em termos nominais (sem desconto de inflação).")

    linha_exibicao = df_exibicao.iloc[-1]
    impostos_exibidos = impostos["total"] / fator_deflator_final
    saldo_liquido_exibido = saldo_liquido / fator_deflator_final

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Investido", formatar_moeda(linha_exibicao["capital_investido"], decimais=0))
    col_b.metric("Saldo Bruto", formatar_moeda(linha_exibicao["saldo_bruto"], decimais=0))
    rotulo_imposto = "Impostos (IR + IOF)" if impostos["iof"] > 0 else "Impostos (IR)"
    col_c.metric(rotulo_imposto, formatar_moeda(impostos_exibidos, decimais=0))
    col_d.metric("Saldo Líquido", formatar_moeda(saldo_liquido_exibido, decimais=0))

    st.plotly_chart(renderizar_grafico_area_empilhada(df_exibicao, eventos), use_container_width=True)

    st.divider()
    if st.button("Gerar Análise com IA"):
        resultados = {
            "total_investido": linha_exibicao["capital_investido"],
            "saldo_bruto": linha_exibicao["saldo_bruto"],
            "impostos": impostos_exibidos,
            "saldo_liquido": saldo_liquido_exibido,
            "taxa_ativo_pct": taxa_anual * 100,
            "taxa_reinvestimento_pct": taxa_reinvestimento_pct,
            "esforco_investidor_pct": esforco_investidor_pct,
            "esforco_tempo_pct": esforco_tempo_pct,
        }
        with st.spinner("Consultando o Gemini..."):
            try:
                texto_analise = gerar_analise_ia(montar_prompt_analise(resultados))
                st.info(texto_analise.replace("$", "\\$"), icon=None)
            except Exception as exc:
                st.error(f"Não foi possível gerar a análise agora: {exc}")


def renderizar_grafico_sensibilidade_pu(
    pu_atual: float, taxa_atual_pct: float, du_uteis: int, delta_taxa_pp: float
) -> go.Figure:
    deltas = np.linspace(-DELTA_TAXA_MAXIMO_PP, DELTA_TAXA_MAXIMO_PP, 61)
    pus_simulados = [
        simular_pu_estressado(pu_atual, taxa_atual_pct, float(delta), du_uteis)[0] for delta in deltas
    ]
    pu_cenario, _ = simular_pu_estressado(pu_atual, taxa_atual_pct, delta_taxa_pp, du_uteis)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=pus_simulados,
            name="PU simulado",
            mode="lines",
            line=dict(width=2.5, color="#173451"),
            hovertemplate="Δ taxa: %{x:.2f} p.p.<br>PU: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[pu_atual],
            name="PU Atual",
            mode="markers",
            marker=dict(size=11, color="#8AA0B9"),
            hovertemplate="PU Atual: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[delta_taxa_pp],
            y=[pu_cenario],
            name="Cenário Simulado",
            mode="markers",
            marker=dict(size=13, color="#F19828", symbol="diamond"),
            hovertemplate="Cenário: Δ %{x:.2f} p.p.<br>PU: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#2D3748"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="closest",
        xaxis=dict(title="Ajuste na taxa (pontos percentuais)", showgrid=False, zeroline=True, zerolinecolor="#DDE3EA"),
        yaxis=dict(title="PU (R$)", showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickprefix="R$ "),
    )
    return fig


def renderizar_marcacao_mercado() -> None:
    st.header(PAGINAS["Marcação a Mercado"]["titulo"])
    st.write(PAGINAS["Marcação a Mercado"]["descricao"])

    sufixo_reset = st.session_state.get(CHAVE_CONTADOR_RESET, 0)

    col1, col2 = st.columns(2)
    with col1:
        tipo_titulo_nome = st.selectbox(
            "Tipo de Título",
            list(TIPOS_TITULO_MARCACAO.keys()),
            key=f"{CHAVE_MM_TIPO_TITULO}_{sufixo_reset}",
        )
    categoria = TIPOS_TITULO_MARCACAO[tipo_titulo_nome]

    if categoria == "selic":
        titulos_selic = get_titulos_selic()
        with col2:
            if not titulos_selic.empty:
                st.caption(
                    f"Spread de compra vigente: {float(titulos_selic['taxa_compra'].iloc[0]):.2f}% "
                    "sobre a Selic."
                )
        st.info(
            "O Tesouro Selic (LFT) tem seu rendimento atrelado à Selic diariamente, então seu "
            "preço acompanha o valor justo quase todo dia — a marcação a mercado praticamente "
            "não gera ágio ou deságio relevante neste título, mesmo em cenários de estresse na "
            "curva de juros. É por isso que ele costuma ser o mais indicado para reserva de "
            "emergência: baixo risco de perda se você precisar vender antes do vencimento."
        )
        return

    titulos = obter_titulos_zero_cupom(categoria)
    if titulos.empty:
        st.warning("Não há títulos sem cupom disponíveis para este tipo no momento.")
        return

    opcoes_rotulos = [
        f"{linha.data_vencimento.strftime('%d/%m/%Y')} — taxa atual: {linha.taxa_compra:.2f}% a.a."
        for linha in titulos.itertuples()
    ]
    with col2:
        indice_escolhido = st.selectbox(
            "Vencimento",
            range(len(opcoes_rotulos)),
            format_func=lambda i: opcoes_rotulos[i],
            key=f"{CHAVE_MM_VENCIMENTO}_{sufixo_reset}_{categoria}",
        )

    titulo_escolhido = titulos.iloc[indice_escolhido]
    pu_atual = float(titulo_escolhido["pu_compra"])
    taxa_atual_pct = float(titulo_escolhido["taxa_compra"])
    du_uteis = calcular_du_uteis(titulo_escolhido["data_vencimento"])
    duration_anos = calcular_duration_anos(du_uteis)

    st.caption(
        f"Faltam {duration_anos:.1f} anos até o vencimento. Como este título não paga cupom, "
        f"a duration (sensibilidade do preço a mudanças na taxa) é igual ao prazo: "
        f"{duration_anos:.1f} anos — quanto maior, maior o efeito de qualquer variação na taxa "
        "sobre o preço. Considera apenas dias úteis, sem contar feriados."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Cenário Otimista: taxa cai", use_container_width=True):
            st.session_state[f"{CHAVE_MM_DELTA_TAXA}_{sufixo_reset}"] = DELTA_CENARIO_OTIMISTA_PP
    with col_btn2:
        if st.button("Cenário de Estresse: taxa sobe", use_container_width=True):
            st.session_state[f"{CHAVE_MM_DELTA_TAXA}_{sufixo_reset}"] = DELTA_CENARIO_ESTRESSE_PP

    delta_taxa_pp = st.slider(
        "Ajuste na taxa deste título (pontos percentuais)",
        min_value=-DELTA_TAXA_MAXIMO_PP,
        max_value=DELTA_TAXA_MAXIMO_PP,
        value=0.0,
        step=0.1,
        key=f"{CHAVE_MM_DELTA_TAXA}_{sufixo_reset}",
    )

    if delta_taxa_pp > 0:
        st.caption(
            f"Se a taxa deste título subir {delta_taxa_pp:.2f} pontos percentuais, o preço cai "
            "(deságio) — quem vender antes do vencimento recebe menos do que o valor aplicado "
            "corrigido pela taxa contratada."
        )
    elif delta_taxa_pp < 0:
        st.caption(
            f"Se a taxa deste título cair {abs(delta_taxa_pp):.2f} pontos percentuais, o preço "
            "sobe (ágio) — quem vender antes do vencimento recebe mais do que o valor aplicado "
            "corrigido pela taxa contratada."
        )
    else:
        st.caption("Ajuste a taxa acima (ou use os botões de cenário) para simular uma venda antecipada.")

    pu_novo, taxa_nova_pct = simular_pu_estressado(pu_atual, taxa_atual_pct, delta_taxa_pp, du_uteis)
    variacao_pct = (pu_novo / pu_atual - 1) * 100
    diferenca_pu = pu_novo - pu_atual
    sinal_diferenca = "+" if diferenca_pu >= 0 else "-"

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("PU Atual", formatar_moeda(pu_atual))
    col_b.metric("Taxa Simulada", f"{taxa_nova_pct:.2f}% a.a.")
    col_c.metric(
        "PU Simulado",
        formatar_moeda(pu_novo),
        delta=f"{sinal_diferenca}{formatar_moeda(abs(diferenca_pu))}",
    )
    col_d.metric("Ágio/Deságio", f"{variacao_pct:+.2f}%")

    st.plotly_chart(
        renderizar_grafico_sensibilidade_pu(pu_atual, taxa_atual_pct, du_uteis, delta_taxa_pp),
        use_container_width=True,
    )

    st.subheader("Comparativo entre vencimentos")
    st.caption(
        f"Todos os vencimentos de {tipo_titulo_nome} sob o mesmo cenário simulado acima "
        f"({delta_taxa_pp:+.2f} p.p.), ordenados do maior para o menor ágio/deságio — "
        "útil para comparar em qual vencimento vale mais a pena comprar se a aposta é ganhar "
        "com a marcação a mercado."
    )
    st.dataframe(
        montar_tabela_sensibilidade(titulos, delta_taxa_pp),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Duration (anos)": st.column_config.NumberColumn(format="%.1f"),
            "Taxa Atual (% a.a.)": st.column_config.NumberColumn(format="%.2f%%"),
            "PU Atual (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "Taxa Simulada (% a.a.)": st.column_config.NumberColumn(format="%.2f%%"),
            "PU Simulado (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "Ágio/Deságio (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    with st.expander("Como funciona a marcação a mercado?"):
        st.write(
            "O preço (PU) de um título prefixado ou IPCA+ sem cupom é o valor de face trazido "
            "a valor presente pela taxa de juros vigente. Se a taxa de mercado sobe depois que "
            "você comprou, o mesmo título passa a valer menos hoje (deságio); se cai, passa a "
            "valer mais (ágio). Isso só importa para quem vende antes do vencimento — quem "
            "carrega o título até o final recebe exatamente a taxa que contratou na compra, "
            "independentemente do que aconteceu no meio do caminho."
        )

    st.divider()
    if st.button("Gerar Análise com IA", key=f"marcacao_btn_ia_{sufixo_reset}"):
        resultados = {
            "pu_atual": pu_atual,
            "pu_novo": pu_novo,
            "variacao_pct": variacao_pct,
            "taxa_atual_pct": taxa_atual_pct,
            "taxa_nova_pct": taxa_nova_pct,
            "delta_taxa_pp": delta_taxa_pp,
            "duration_anos": duration_anos,
            "prazo_restante_anos": duration_anos,
        }
        with st.spinner("Consultando o Gemini..."):
            try:
                texto_analise = gerar_analise_ia(montar_prompt_analise_marcacao(resultados))
                st.info(texto_analise.replace("$", "\\$"), icon=None)
            except Exception as exc:
                st.error(f"Não foi possível gerar a análise agora: {exc}")


def renderizar_grafico_baldes_cascata(df: pd.DataFrame, mes_atingido: dict, nomes: dict) -> go.Figure:
    cores = {"curto": "#F19828", "medio": "#8AA0B9", "longo": "#173451"}
    fig = go.Figure()
    for chave in ORDEM_BALDES:
        fig.add_trace(
            go.Scatter(
                x=df["mes"],
                y=df[f"saldo_{chave}"],
                name=nomes[chave],
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=cores[chave]),
                fillcolor=cores[chave],
                hovertemplate=f"Mês %{{x}}<br>{nomes[chave]}: R$ %{{y:,.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#2D3748"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        xaxis=dict(title="Meses", showgrid=False, zeroline=False),
        yaxis=dict(title="Valor (R$)", showgrid=True, gridcolor="#E2E8F0", zeroline=False, tickprefix="R$ "),
    )

    posicoes_anotacao = ["top left", "top right"]
    chaves_com_marco = [chave for chave in ORDEM_BALDES[:-1] if mes_atingido[chave] is not None]
    for indice, chave in enumerate(chaves_com_marco):
        fig.add_vline(
            x=mes_atingido[chave],
            line_dash="dash",
            line_color="#2D3748",
            annotation_text=f"{nomes[chave]}: meta atingida",
            annotation_position=posicoes_anotacao[indice % len(posicoes_anotacao)],
            annotation_font_color="#2D3748",
            annotation_font_size=11,
        )

    return fig


def renderizar_objetivos_financeiros() -> None:
    st.header(PAGINAS["Objetivos Financeiros"]["titulo"])
    st.write(PAGINAS["Objetivos Financeiros"]["descricao"])

    sufixo_reset = st.session_state.get(CHAVE_CONTADOR_RESET, 0)

    try:
        _, cdi_pct_atual = obter_selic_cdi_atuais()
    except Exception:
        cdi_pct_atual = None

    col1, col2 = st.columns(2)
    with col1:
        aporte_mensal_total = st.number_input(
            "Aporte Mensal Total Disponível (R$)",
            min_value=0.0,
            value=1000.0,
            step=50.0,
            key=f"{CHAVE_OF_APORTE_MENSAL_TOTAL}_{sufixo_reset}",
        )
    with col2:
        taxa_esperada_padrao = cdi_pct_atual if cdi_pct_atual is not None else TAXA_REINVESTIMENTO_PADRAO_PCT
        taxa_esperada_pct = st.number_input(
            "Taxa de Rendimento Esperada (% a.a.)",
            min_value=0.0,
            value=taxa_esperada_padrao,
            step=0.1,
            key=f"{CHAVE_OF_TAXA_ESPERADA}_{sufixo_reset}",
        )

    st.caption(
        f"Distribuição inicial do aporte: {FATIA_INICIAL_BALDES['curto'] * 100:.0f}% Curto Prazo, "
        f"{FATIA_INICIAL_BALDES['medio'] * 100:.0f}% Médio Prazo, {FATIA_INICIAL_BALDES['longo'] * 100:.0f}% "
        "Longo Prazo. Quando um objetivo é atingido, a fatia dele é redirecionada para acelerar os "
        "objetivos seguintes (efeito cascata)."
    )

    col_curto, col_medio, col_longo = st.columns(3)
    with col_curto:
        st.markdown("**Curto Prazo**")
        nome_curto = st.text_input(
            "Nome do Objetivo", value="Reserva de Emergência", key=f"{CHAVE_OF_NOME_CURTO}_{sufixo_reset}"
        )
        valor_curto = st.number_input(
            "Valor Alvo (R$)", min_value=100.0, value=10000.0, step=500.0, key=f"{CHAVE_OF_VALOR_CURTO}_{sufixo_reset}"
        )
    with col_medio:
        st.markdown("**Médio Prazo**")
        nome_medio = st.text_input(
            "Nome do Objetivo", value="Entrada do Imóvel", key=f"{CHAVE_OF_NOME_MEDIO}_{sufixo_reset}"
        )
        valor_medio = st.number_input(
            "Valor Alvo (R$)", min_value=100.0, value=50000.0, step=1000.0, key=f"{CHAVE_OF_VALOR_MEDIO}_{sufixo_reset}"
        )
    with col_longo:
        st.markdown("**Longo Prazo**")
        nome_longo = st.text_input(
            "Nome do Objetivo", value="Aposentadoria", key=f"{CHAVE_OF_NOME_LONGO}_{sufixo_reset}"
        )
        valor_longo = st.number_input(
            "Valor Alvo (R$)", min_value=100.0, value=300000.0, step=5000.0, key=f"{CHAVE_OF_VALOR_LONGO}_{sufixo_reset}"
        )

    if aporte_mensal_total <= 0:
        st.warning("Informe um aporte mensal maior que zero para simular os objetivos.")
        return

    nomes = {"curto": nome_curto, "medio": nome_medio, "longo": nome_longo}
    valores_alvo = {"curto": valor_curto, "medio": valor_medio, "longo": valor_longo}

    df, mes_atingido = simular_baldes_cascata(aporte_mensal_total, taxa_esperada_pct / 100, valores_alvo)

    tempos = {chave: formatar_anos_meses(mes_atingido[chave]) for chave in ORDEM_BALDES}

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(nome_curto, tempos["curto"])
    col_b.metric(nome_medio, tempos["medio"])
    col_c.metric(nome_longo, tempos["longo"])

    if any(mes_atingido[chave] is None for chave in ORDEM_BALDES):
        st.warning(
            f"Pelo menos um objetivo não foi atingido em {LIMITE_MESES_SEGURANCA_BALDES // 12} anos com "
            "essas premissas. Considere aumentar o aporte mensal ou reduzir os valores-alvo."
        )

    st.plotly_chart(renderizar_grafico_baldes_cascata(df, mes_atingido, nomes), use_container_width=True)

    st.divider()
    if st.button("Gerar Análise com IA", key=f"objetivos_btn_ia_{sufixo_reset}"):
        resultados = {
            "aporte_mensal_total": aporte_mensal_total,
            "taxa_esperada_pct": taxa_esperada_pct,
            "nome_curto": nome_curto,
            "valor_curto": valor_curto,
            "tempo_curto": tempos["curto"],
            "nome_medio": nome_medio,
            "valor_medio": valor_medio,
            "tempo_medio": tempos["medio"],
            "nome_longo": nome_longo,
            "valor_longo": valor_longo,
            "tempo_longo": tempos["longo"],
        }
        with st.spinner("Consultando o Gemini..."):
            try:
                texto_analise = gerar_analise_ia(montar_prompt_analise_baldes(resultados))
                st.info(texto_analise.replace("$", "\\$"), icon=None)
            except Exception as exc:
                st.error(f"Não foi possível gerar a análise agora: {exc}")


with st.sidebar:
    if st.button("Reiniciar Premissas", use_container_width=True):
        st.session_state[CHAVE_CONTADOR_RESET] = st.session_state.get(CHAVE_CONTADOR_RESET, 0) + 1
        st.rerun()

    st.markdown("### Simulador de Renda Fixa")
    pagina_selecionada = st.radio(
        "Navegação",
        list(PAGINAS.keys()),
        label_visibility="collapsed",
    )

if pagina_selecionada == "Valor Futuro":
    renderizar_valor_futuro()
elif pagina_selecionada == "Marcação a Mercado":
    renderizar_marcacao_mercado()
elif pagina_selecionada == "Objetivos Financeiros":
    renderizar_objetivos_financeiros()
else:
    pagina = PAGINAS[pagina_selecionada]
    st.header(pagina["titulo"])
    st.write(pagina["descricao"])
