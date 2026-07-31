"""Aplicativo principal: casca visual + Feature 1 (Projeção de Valor Futuro) do Simulador de Renda Fixa."""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai

from data_fetch import (
    get_cotacoes_etfs,
    get_curva_prefixada,
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
        "titulo": "Simulador de Objetivos Financeiros (Acumulação)",
        "descricao": (
            "Aqui o usuário vai projetar a acumulação de patrimônio ao longo "
            "do tempo, com aporte inicial, aportes mensais e opção de step-up, "
            "separando capital investido e juros compostos."
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

TAXA_PADRAO_FALLBACK = 0.10
TAXA_INFLACAO_APORTES = 0.045
TAXA_REINVESTIMENTO_PADRAO_PCT = 8.5
TAXA_POUPANCA_MENSAL_REGRA_ALTA = 0.005
LIMIAR_SELIC_POUPANCA_PCT = 8.5
PERCENTUAL_POUPANCA_SOBRE_SELIC = 0.7
DIAS_UTEIS_ANO = 252
PRAZO_MAXIMO_ANOS = 30

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


def gerar_analise_ia(resultados: dict) -> str:
    """Chama o Gemini para gerar uma análise textual curta do resultado da simulação."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não configurada no .env.")

    cliente = genai.Client(api_key=api_key)
    resposta = cliente.models.generate_content(
        model=MODELO_GEMINI,
        contents=montar_prompt_analise(resultados),
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
                texto_analise = gerar_analise_ia(resultados)
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
else:
    pagina = PAGINAS[pagina_selecionada]
    st.header(pagina["titulo"])
    st.write(pagina["descricao"])
