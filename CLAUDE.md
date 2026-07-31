# Projeto: Simulador Avançado de Renda Fixa e Estratégia de Alocação

## 1. Visão Geral
Este é um aplicativo em Python desenvolvido com Streamlit focado em simulações financeiras avançadas de Renda Fixa. O objetivo é traduzir conceitos complexos (como marcação a mercado e inflação implícita) em uma interface limpa, didática e de fácil uso para investidores leigos. O projeto consome dados reais de mercado e utiliza uma IA Generativa para criar análises textuais personalizadas dos resultados.

## 2. Stack Tecnológico
*   **Front-end & Framework:** Streamlit (Estritamente SEM EMOJIS em toda a interface).
*   **Visualização de Dados:** Plotly (Gráficos interativos com Storytelling with Data).
*   **Integração de Dados:** `tradingcomdados` ou `bcb` (API do Banco Central) para curva DI e Tesouro Direto; `yfinance` para cotações de ETFs.
*   **LLM API:** Google Gemini (chave em `.env`, variável `GEMINI_API_KEY`).
*   **Manipulação de Dados:** Pandas e NumPy.

## 3. Diretrizes de Design e UX/UI (Strict)
*   **Tipografia:** Inter, Roboto ou Open Sans (sem serifa, alta legibilidade para numerais).
*   **Cores Principais:**
    *   Azul Escuro (Primária / Base Gráficos): `#173451`
    *   Laranja Vibrante (Destaque / Ação / Insights): `#F19828`
*   **Cores de Fundo e Texto:**
    *   Background Tela: `#DDE3EA` ou `#E2E8F0`
    *   Fundo Cards/Widgets: `#FFFFFF`
    *   Texto Principal (Títulos/Eixos): `#2D3748`
    *   Texto Secundário (Descrições/Menus inativos): `#8AA0B9`
*   **Regra de Ouro Visual:** O laranja vibrante (`#F19828`) não é decorativo. Deve ser usado estritamente para destacar a curva de juros no gráfico, os botões de ação ou os KPIs de resultado final. O restante da interface deve manter neutralidade com o azul escuro e os cinzas.

## 4. Features Principais (As 5 Abas do Simulador)

### Feature 1: Projeção de Valor Futuro (Tesouro Direto vs. ETFs de RF)
*   **Inputs:** Aporte inicial, aportes mensais, prazo de investimento (slider).
*   **Ativos:** Tesouro Direto (Taxas extraídas em tempo real) e ETFs de Renda Fixa (AUPO11, LFTS11, LFTI11, NCDI11).
*   **Lógica Tributária:**
    *   Tesouro Direto: Tabela Regressiva Padrão (22,5% a 15%) + IOF nos primeiros 30 dias.
    *   ETFs de Renda Fixa: Alíquota fixa de 15% sobre o lucro (independente do prazo) e isenção de come-cotas.

### Feature 2: Simulador de Marcação a Mercado
*   **Objetivo:** Mostrar o impacto (ágio/deságio) se o usuário resgatar o título antes do vencimento mediante cenários de estresse na curva de juros.
*   **UX:** Utilizar termos acessíveis. Ex: "Cenário Otimista: Se a taxa cair X%" ao invés de "Fechamento da Curva DI".
*   **Cálculo:** Requer o cálculo do Preço Unitário (PU) atualizado com base na *duration* do título escolhido versus a nova taxa simulada.

### Feature 3: Simulador de Objetivos Financeiros (Acumulação)
*   **Inputs:** Aporte inicial, aporte mensal, taxa de remuneração esperada, opção de *step-up* (aumentar aportes em % em um ano específico).
*   **Output Gráfico:** Gráfico de área (Plotly) exibindo a área de Capital Investido (Azul Escuro) empilhada com a área de Juros Compostos Acumulados (Laranja Vibrante).

### Feature 4: Análise Generativa (LLM)
*   Para cada simulação realizada nas Features 1, 2, 3 e 5, o sistema fará a matemática no back-end e injetará os resultados em um prompt estruturado.
*   O LLM retornará um texto explicativo, de no máximo 150 palavras, traduzindo o resultado de forma simples, direta e didática. Nenhuma lógica pesada de `if/else` deve ser usada para montar a frase.

### Feature 5: Conta Inversa (Planejamento Previdenciário/Independência)
*   **Inputs:** Valor alvo futuro (em termos reais), aporte inicial, opções de taxas de remuneração.
*   **Output:** Cálculo algébrico do aporte mensal necessário para atingir o alvo no prazo estipulado.

## 5. Regras Matemáticas Vitais
*   **Inflação Implícita (Break-even Inflation):** A premissa de inflação não será chutada pelo usuário. O sistema calculará a inflação implícita do mercado baseada nos títulos públicos correntes, utilizando a equação de Fisher e evidenciando a metodologia na interface (Tooltip ou Expander).
*   A relação matemática que deve ser programada é:
$$ \text{Inflação Implícita} = \left( \frac{1 + \text{Taxa Pré-fixada}}{1 + \text{Taxa Real (IPCA+)}} \right) - 1 $$

## 6. Etapas de Execução Sugeridas
1.  **Etapa de Infraestrutura:** Configurar ambiente, criar `requirements.txt`, estruturar as chaves da API (`.env`).
2.  **Etapa de Dados (Data Layer):** Construir as funções de requisição e tratamento de dados das bibliotecas (`bcb`, `tradingcomdados`, `yfinance`) para garantir que as taxas estejam funcionando e armazenadas em cache (`@st.cache_data`).
3.  **Etapa Visual e de Layout:** Montar o esqueleto do Streamlit (as 5 abas) e aplicar o arquivo CSS/Theme customizado com os *Hex Codes*.
4.  **Etapa de Modelagem Matemática:** Programar e validar os cálculos financeiros (VPL, Marcação a Mercado, Impostos).
5.  **Etapa Gráfica e LLM:** Integrar os gráficos via Plotly e estruturar a requisição à API (OpenAI/Gemini) para os textos finais.