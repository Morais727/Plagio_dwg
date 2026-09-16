# Detector de Possível Plágio em Desenhos Técnicos CAD (DWG/DXF)

Sistema em Python para analisar arquivos DWG/DXF e identificar pares suspeitos de cópia, gerando um índice de suspeita explicável (0–100) para revisão pelo professor.

## Status

| Módulo | Status |
|---|---|
| `config.py` | ✅ |
| `detector/reader.py` | ✅ |
| `detector/normalizer.py` | ✅ |
| `detector/features.py` | ✅ |
| `detector/graph.py` | ✅ |
| `detector/sequence.py` | ✅ |
| `detector/similarity.py` | ✅ |
| `detector/dwg_converter.py` | ✅ |
| `detector/report.py` | ✅ |
| `detector/classifier.py` | ⏳ (opcional) |
| `ui/main_window.py` | ✅ |
| Testes | ✅ (161) |

## Instalação

### Dependências Python

```bash
pip install -r requirements.txt
```

### ODA File Converter (para leitura DWG)

O ODA File Converter é um programa externo gratuito (para uso acadêmico) que converte DWG para DXF.

```bash
python scripts/install_oda.py
```

No Linux, a conversão usa `xvfb-run` para suprimir a janela GUI.

## Uso

### Interface gráfica

```bash
python app.py
```

Selecionar pasta com arquivos `.dxf` — o processamento inicia automaticamente.

### Linha de comando

```bash
python app.py --folder data/meus_desenhos
```

### Testes

```bash
pytest
pytest --cov=detector --cov-report=term-missing
```

## Configuração

Edite `config.py` para ajustar:
- Pesos de cada componente do score
- Número de bins dos histogramas
- Tolerâncias geométricas
- Caminho do ODA File Converter
- Versão de saída do DXF

## Score

| Componente | Peso | Descrição |
|---|---|---|
| Geometria | 35% | Similaridade de features geométricas |
| Sequência | 35% | Ordem das entidades no arquivo |
| Grafo | 20% | Relações topológicas entre entidades |
| Estilos/Layers | 10% | Layers, blocos, estilos |

Faixas de cor:
- < 70% → verde
- 70–80% → amarelo
- 80–95% → laranja
- > 95% → vermelho
- 100% → exibe "cópia"

## Pipeline

```
Arquivos DXF → Reader → Normalizer → FeatureExtractor → SimilarityEngine → Score 0-100
                                    → GraphBuilder       → 
                                    → SequenceAnalyzer   → 
                                                        → ReportGenerator → PDF/HTML
```

## Módulos

### Reader (`detector/reader.py`)
Lê arquivos DXF com `ezdxf`. Extrai entidades (LINE, ARC, CIRCLE, LWPOLYLINE, INSERT, TEXT, MTEXT, DIMENSION), layers, blocos, estilos de texto/cota e metadados (autor, versão, timestamps). Retorna um `CadDocument` imutável.

### DWG Converter (`detector/dwg_converter.py`)
Converte arquivos DWG para DXF usando ODA File Converter. Usa `xvfb-run` no Linux para suprimir a janela GUI. Config path cross‑platform (Linux: `~/.config/plagio_dwg`, Windows: `%APPDATA%/plagio_dwg`).

### Normalizer (`detector/normalizer.py`)
Remove diferenças irrelevantes de posição (centraliza na origem), rotação (alinha eixos principais via PCA) e escala (normaliza bounding box).

### FeatureExtractor (`detector/features.py`)
Produz um `FeatureVector` imutável com contagens por tipo de entidade, histograma de ângulos, histograma de comprimentos, bounding box, densidade espacial, uso de layers, uso de blocos e precisão decimal.

### GraphBuilder (`detector/graph.py`)
Constrói um grafo NetworkX onde cada nó é uma entidade e as arestas representam relações geométricas (interseção, paralelismo, perpendicularidade, tangência). Extrai métricas: degree centrality, betweenness centrality, clustering coefficient, componentes.

### SequenceAnalyzer (`detector/sequence.py`)
Converte a sequência de entidades em string de códigos (L, A, C, P, I, T, M, D). Compara pares usando LCS e distância Levenshtein via `RapidFuzz`. Suporta DTW como opção.

### SimilarityEngine (`detector/similarity.py`)
Combina os componentes em score 0–100 com pesos configuráveis. Cada componente produz score individual.

### ReportGenerator (`detector/report.py`)
Gera relatório para cada par em **HTML** e **PDF** com índice de suspeita, score por componente, tabela de métricas e imagens lado a lado.

### GUI (`ui/main_window.py`)
Interface PySide6 com seleção de pasta (processamento automático), tabela ranking ordenada, diálogo de comparação lado a lado e exportação de relatório PDF.

## Tecnologias

Python 3.12+, ezdxf, NumPy, NetworkX, pandas, RapidFuzz, matplotlib, PySide6.

## Licença

MIT
