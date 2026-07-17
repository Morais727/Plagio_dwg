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
└── LICENSE
```

## Tecnologias

Python 3.12+, ezdxf, NumPy, SciPy, NetworkX, pandas, RapidFuzz, matplotlib, PySide6.

## Como usar (em breve)

```bash
pip install -r requirements.txt
python app.py
```

## Licença

MIT
