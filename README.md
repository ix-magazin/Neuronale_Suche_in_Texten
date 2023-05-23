# Neuronale_Suche_in_Texten
Git-Hub-Verzeichnis zum [Wissens-Artikel](https://www.heise.de/select/ix/2023/6/2308009005309487505) von Leon Marius Schröder, Clemens Gutknecht und Leon Lukas, erschienen auf [Heise+](https://heise.de/-8992144) und im [iX-Magazin 06/2023](https://www.heise.de/select/ix/2023/6).

# iX-tract
- Systeme zum Durchsuchen von Dokumentsammlungen basieren häufig auf dem statistischen Maß TF-IDF, mit dem sich jedoch keine semantischen Verbindungen in Dokumenten finden lassen. Die Lösung: neuronale Suchmaschinen.
- Dense Passage Retrieval (DPR), basierend auf Sprachmodellen, kann semantische Zusammenhänge durch Vektor-Embeddings abbilden und sucht ebenfalls in Form von Embeddings.
- Dank der verfügbaren Sprachmodelle im Hugging-Face-Transformers-Paket und Frage-und-Antwort-Datensätzen wie GermanDPR und GermanQuAD lassen sich einfache neuronale DPR-Modelle schnell in PyTorch umsetzen.
- Um ein allgemeines neuronales DPR-Modell auf das eigene Fachvokabular zu spezialisieren, erstellt etwa das Haystack  Annotation Tool domänenspezifische Frage-und-Antwort-Datensätze zum Trainieren des Modells.
___

**Anforderungen**: Um selbst zu trainieren, wird mindestens eine CUDA-fähige Grafikkarte benötigt.

## Vorbereitung der Arbeitsumgebung:

1. [Miniconda installieren](https://docs.conda.io/en/latest/miniconda.html)
2. [PyTorch installieren](https://pytorch.org/get-started/locally/) (im Konfigurator als Package "Conda" anwählen)
3. [Hugging Face Transformers installieren](https://huggingface.co/docs/transformers/installation#install-with-conda)
4. [Hugging Face Tokenizers installieren](https://huggingface.co/docs/tokenizers/installation)
5. [Hugging Face Datasets installieren](https://huggingface.co/docs/tokenizers/installation)
6. [Haystack installieren](https://docs.haystack.deepset.ai/docs/installation)
7. [Elasticsearch installieren](https://www.elastic.co/downloads/elasticsearch)
8. [sklearn installieren](https://scikit-learn.org/stable/install.html)

Der lokale Dateipfad zu diesem Repository muss zur PYTHONPATH-Umgebungsvariable hinzugefügt werden.

## Schritt für Schritt

Nun können die Scripte nacheinander angepasst und ausgeführt werden.
