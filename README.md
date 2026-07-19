# Detector de Possível Plágio em Desenhos Técnicos CAD (DWG/DXF)

Sistema em Python para analisar arquivos DWG/DXF produzidos em uma disciplina de desenho técnico e identificar pares suspeitos de cópia, gerando um índice de suspeita explicável (0–100). O sistema **não** acusa plágio automaticamente — apenas fornece evidências para revisão pelo professor.

## Pré-requisitos

- **Python 3.12+**
- **ODA File Converter** (gratuito para uso acadêmico) — para converter arquivos `.dwg` em `.dxf`
  - Download: https://www.opendesign.com/guestfiles/oda_file_converter
- **xvfb** (opcional, recomendado) — para o ODA rodar sem abrir janela gráfica
  - `sudo apt install xvfb`

## Instalação

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd plagio_dwg

# 2. Criar ambiente virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependências Python
pip install -r requirements.txt

# 4. Configurar o ODA File Converter
python scripts/install_oda.py
```

O script `install_oda.py` irá:
- Verificar se o ODA File Converter já está instalado
- Abrir o navegador no site de download (se necessário)
- Salvar o caminho do executável em `~/.config/plagio_dwg/oda_path`

Para instalar o `xvfb` (elimina a janela gráfica do ODA):
```bash
sudo apt install xvfb
```

## Uso

### Interface gráfica

```bash
python app.py
```

## Estrutura do projeto

```
project/
├── app.py                    # ponto de entrada da GUI
├── config.py                 # configuração centralizada (pesos, paths, etc.)
├── detector/
│   ├── dwg_converter.py      # conversão DWG → DXF via ODA File Converter
│   ├── reader.py             # leitura de arquivos DXF/DWG
│   ├── normalizer.py         # normalização geométrica
│   ├── features.py           # extração de características
│   ├── graph.py              # construção de grafo (NetworkX)
│   ├── sequence.py           # análise de sequência
│   ├── similarity.py         # engine de similaridade
│   ├── classifier.py         # ML supervisionado (opcional)
│   └── report.py             # geração de relatórios PDF/HTML
├── ui/
│   └── main_window.py        # interface PySide6
├── scripts/
│   └── install_oda.py        # script de instalação/configuração do ODA
├── tests/                    # testes unitários e de integração
├── data/                     # arquivos de entrada (DWG/DXF)
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

## Tecnologias

- Python 3.12+
- ezdxf (leitura DXF)
- NumPy (álgebra linear)
- NetworkX (grafos)
- pandas (tabelas)
- RapidFuzz (comparação de sequências)
- matplotlib (visualizações)
- PySide6 (GUI)
- ODA File Converter (conversão DWG → DXF)
- pytest (testes)

## Licença

MIT
