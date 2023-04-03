# DPR Explained - Dense Passage Retrieval Schritt für Schritt

*Anforderungen*: Um selbst zu trainieren, wird mindestens eine CUDA-fähige Grafikkarte benötigt.

## Vorbereitung der Arbeitsumgebung:

1. [Miniconda installieren](https://docs.conda.io/en/latest/miniconda.html)
2. [PyTorch installieren](https://pytorch.org/get-started/locally/) (im Konfigurator als Package "Conda" anwählen)
3. [huggingface transformers installieren](https://huggingface.co/docs/transformers/installation#install-with-conda)
4. [huggingface tokenizers installieren](https://huggingface.co/docs/tokenizers/installation)
5. [huggingface datasets installieren](https://huggingface.co/docs/tokenizers/installation)
6. [Haystack installieren](https://docs.haystack.deepset.ai/docs/installation)
7. [Elasticsearch installieren](https://www.elastic.co/downloads/elasticsearch)
8. [sklearn installieren](https://scikit-learn.org/stable/install.html)

Der lokale Dateipfad zu diesem Repository muss zur PYTHONPATH-Umgebungsvariable hinzugefügt werden.

## Schritt für Schritt

Nun können die Scripte nacheinander angepasst und ausgeführt werden.