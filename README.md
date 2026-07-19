# Detector de Possível Plágio em Desenhos Técnicos CAD (DWG/DXF)

Sistema em Python para analisar arquivos DWG/DXF e identificar pares suspeitos de cópia, gerando um índice de suspeita explicável (0–100) para revisão pelo professor.

## Status

| Módulo | Status |
|---|---|
| `config.py` | ✅ |
| `detector/reader.py` | ✅ |
| `detector/normalizer.py` | ✅ |
| `detector/features.py` | ⏳ |
| `detector/graph.py` | ⏳ |
| `detector/sequence.py` | ⏳ |
| `detector/similarity.py` | ⏳ |
| `detector/report.py` | ⏳ |
| `detector/classifier.py` | ⏳ |
| `ui/main_window.py` | ⏳ |
| `app.py` | ⏳ |
| Testes | ⏳ |

## Estrutura

```
project/
├── app.py                  # ponto de entrada (a implementar)
├── config.py               # configuração centralizada
├── detector/
│   ├── reader.py           # leitura DXF (✅)
│   ├── normalizer.py       # normalização geométrica (✅)
│   ├── features.py         # extração de features (⏳)
│   ├── graph.py            # grafo NetworkX (⏳)
│   ├── sequence.py         # análise de sequência (⏳)
│   ├── similarity.py       # engine de similaridade (⏳)
│   ├── classifier.py       # ML opcional (⏳)
│   └── report.py           # relatórios PDF/HTML (⏳)
├── ui/
│   └── main_window.py      # GUI PySide6 (⏳)
├── tests/
│   ├── test_reader.py      # (✅)
│   ├── test_normalizer.py  # (✅)
│   └── ...                 # (⏳)
├── data/                   # arquivos de entrada
├── requirements.txt
└── README.md
```

### Pipeline de análise

1. **Importar** — Converte DWG → DXF via ODA File Converter (quando necessário)
2. **Ler** — Extrai entidades CAD do DXF com `ezdxf`
3. **Normalizar** — Remove efeitos de translação, escala e rotação
4. **Extrair features** — Gera vetor de características geométricas, de estilo e metadados
5. **Comparar** — Compara todos os pares usando múltiplas métricas
6. **Calcular score** — Combina métricas em um índice de suspeita (0–100)

### Componentes do score

| Componente | Peso | Descrição |
|---|---|---|
| Geometria | 35% | Similaridade de features geométricas (ângulos, comprimentos, bounding box, etc.) |
| Sequência | 35% | Ordem das entidades no arquivo (LCS, Levenshtein, DTW) |
| Grafo | 20% | Relações topológicas entre entidades (paralelismo, perpendicularidade, interseções) |
| Estilos/Layers | 10% | Layers, blocos, estilos de texto e cota |

## Configuração

Edite `config.py` para ajustar:
- Pesos de cada componente do score
- Número de bins dos histogramas
- Tolerâncias geométricas
- Caminho do ODA File Converter
- Versão de saída do DXF

## Testes

```bash
# Executar todos os testes
pytest

# Com cobertura
pytest --cov=detector --cov-report=term-missing
```

## Pipeline

```
Arquivos DXF → Reader → Normalizer → FeatureExtractor → SimilarityEngine → Score 0-100
                                    → GraphBuilder       → 
                                    → SequenceAnalyzer   → 
                                                        → ReportGenerator → PDF/HTML
```

## Módulos

### Reader (`detector/reader.py`)
Lê arquivos DXF com `ezdxf`. Extrai entidades (LINE, ARC, CIRCLE, LWPOLYLINE, INSERT, TEXT, MTEXT, DIMENSION), layers, blocos, estilos de texto/cota e metadados (autor, versão, timestamps). Retorna um `CadDocument` imutável com todos os dados.

### Normalizer (`detector/normalizer.py`)
Remove diferenças irrelevantes de posição (centraliza na origem), rotação (alinha eixos principais via PCA) e escala (normaliza bounding box para tamanho unitário). Aplica a transformação a todas as coordenadas, comprimentos e ângulos das entidades.

### FeatureExtractor (`detector/features.py`)
Produz um `FeatureVector` imutável com:
- **Contagem por tipo** de entidade (LINE, ARC, CIRCLE, etc.)
- **Histograma de ângulos** (36 bins, 0–360°)
- **Histograma de comprimentos** (20 bins)
- **Bounding box** (largura, altura, área)
- **Densidade espacial** (grade 10×10)
- **Uso de layers** (quantidade por layer)
- **Uso de blocos** (quantidade por bloco)
- **Precisão decimal** (histograma de casas decimais usadas)

### GraphBuilder (`detector/graph.py`)
Constrói um grafo NetworkX onde cada nó é uma entidade e as arestas representam relações geométricas: interseção, paralelismo, perpendicularidade (entre segmentos) e tangência (entre círculos/arcos e segmentos). Extrai métricas: degree centrality, betweenness centrality, clustering coefficient, número de componentes.

### SequenceAnalyzer (`detector/sequence.py`)
Converte a sequência de entidades do arquivo em uma string de códigos (L, A, C, P, I, T, M, D). Compara pares de sequências usando LCS (Longest Common Subsequence) e distância Levenshtein via `RapidFuzz`. Suporta DTW (Dynamic Time Warping) como opção.

### SimilarityEngine (`detector/similarity.py`)
Combina 5 componentes em um score de 0 a 100 com pesos configuráveis:

| Componente | Peso | Métrica |
|---|---|---|
| Geometria | 35% | Similaridade de histogramas, bounding box, contagens |
| Sequência | 25% | LCS + Levenshtein normalizados |
| Grafo | 20% | Densidade, centralidades, componentes |
| Estilos/Layers | 10% | Jaccard entre conjuntos de layers, blocos, estilos |
| Metadados | 10% | Autor, versão DXF, last saved by |

Cada componente produz um score individual com justificativa textual.

### ReportGenerator (`detector/report.py`)
Gera relatório para cada par comparado contendo: índice de suspeita, score por componente, justificativa textual, tabela de métricas e imagens dos desenhos lado a lado (matplotlib). Exporta em **HTML** (visualização no navegador) e **PDF** (impressão).

### GUI (`ui/main_window.py`)
Interface PySide6 com:
- Botão **Selecionar Pasta** — escolhe diretório com arquivos .dxf
- Botão **Processar** — executa o pipeline completo em thread separada (não trava a界面)
- **Barra de progresso** com status em tempo real
- **Tabela ranking** — pares ordenados do maior score para o menor
- **Clique em um par** → abre diálogo de comparação lado a lado com imagem, métricas e justificativa
- **Botão Exportar PDF** — salva relatório individual

### app.py
Ponto de entrada. Inicializa a aplicação Qt e abre a janela principal.

## Tecnologias

Python 3.12+, ezdxf, NumPy, SciPy, NetworkX, pandas, RapidFuzz, matplotlib, PySide6.

## Como usar (em breve)

```bash
pip install -r requirements.txt
python app.py
```

## Licença

MIT
