import os
import datasets
from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

# Hier bitte den Pfad zu den Dokumenten eintragen, die durchsucht werden sollen.
# Es werden nur Textdateien eingelesen, welche die Dateiendung ".txt" aufweisen.
# PATH_TO_DOCUMENTS = "/pfad/zu/dokumenten"
PATH_TO_DOCUMENTS = "/home/ma/s/schroederl/XNEXT/xnext/data/Staedte_Dataset"

# Hier bitte den Pfad eintragen, unter welchem der Trainings-Datensatz abgespeichert werden soll
# PATH_TO_DATASET = "/pfad/zu/datensatz"
PATH_TO_DATASET = "/home/ma/s/schroederl/DPR-Explained/dataset"

# Der Pfad zum Datensatz-Script wird automatisch ermittelt
PATH_TO_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PATH_TO_SCRIPT = os.path.join(PATH_TO_DIRECTORY, "dprexplained", "passages_dataset.py")

# In der Form bitte nur für einen lokalen Development-Server nutzen. Auf keinen Fall Credentials in Produktivcode
# schreiben.
ELASTIC_HOST = "localhost"
ELASTIC_USERNAME = ""
ELASTIC_PASSWORD = ""
ELASTIC_DOCUMENT_INDEX = "demo_documents"
ELASTIC_PASSAGE_INDEX = "demo_passages"

# Hier muss der DPR Passage Encoder spezifiziert werden (auch Context Encoder genannt). Kann auch ein lokaler Pfad sein.
BASE_DPR_MODEL_NAME_OR_PATH = "deepset/gbert-base-germandpr-ctx_encoder"

# Hier wird spezifiziert, wie groß Passagen sein sollen.
PASSAGE_LENGTH_IN_TOKENS = 200
MIN_PASSAGE_LENGTH_IN_TOKENS = 20

# Hier wird spezifiziert, wie viele Schlüsselworte an die Passagen angehängt werden sollen
NUM_KEYWORDS_PER_DOCUMENT = 10


def main():
    # ACHTUNG: Dieses Script ist für Lernzwecke gedacht. Es skaliert nicht für beliebig große Datenmengen und beinhaltet
    # keinerlei Security-Mechanismen.

    print(f"Lade Dokumente von {PATH_TO_DOCUMENTS} und schreibe sie nach Elasticsearch(host={ELASTIC_HOST}, "
          f"user={ELASTIC_HOST}, "
          f"password={ELASTIC_PASSWORD}) und in das huggingface-Dataset {PATH_TO_DATASET}.")

    document_store = ElasticsearchDocumentStore(host=ELASTIC_HOST,
                                                username=ELASTIC_USERNAME,
                                                password=ELASTIC_PASSWORD,
                                                index=ELASTIC_DOCUMENT_INDEX)

    passage_store = ElasticsearchDocumentStore(host=ELASTIC_HOST,
                                               username=ELASTIC_USERNAME,
                                               password=ELASTIC_PASSWORD,
                                               index=ELASTIC_PASSAGE_INDEX)

    tokenizer = AutoTokenizer.from_pretrained(BASE_DPR_MODEL_NAME_OR_PATH)

    # Hier werden die Text-Dokumente eingelesen, um sie später nach Elasticsearch und in ein huggingface-Dataset
    # zu schreiben. Wer sehr viele Dokumente hat, sodass sie nicht auf einmal in den Speicher passen, muss an der Stelle
    # den Code umschreiben, sodass zwischendurch batch-weise nach Elasticsearch geschrieben wird.
    # Die Dokumente werden zudem in Passagen eingeteilt, damit bei sehr langen Dokumenten einzelne Passagen gefunden
    # werden können.
    # Damit die Passagen wiederum ihren Kontext nicht verlieren, werden mittels TF-IDF Schlüsselworte aus dem Gesamt-
    # Dokument extrahiert und an alle Passagen angehängt.
    document_file_paths = []
    document_texts = []
    for root, _, file_names in os.walk(PATH_TO_DOCUMENTS):
        # Iterate over all files in the current sub-folder
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            # Check if the file is a text file
            if file_path.endswith(".txt"):
                with open(file_path, "r") as f:
                    text = f.read()
                # die Dokument-Texte werden zunächst im Speicher gesammelt
                document_texts.append(text)
                document_file_paths.append(file_path)

    # Die Dokumente liegen nun alle vor. Es können TF-IDF Scores und Schlagworte pro Dokument berechnet werden.
    # Zudem werden die Dokumente nach Elasticsearch geschrieben
    print("Berechne IDF Gewichte ...")
    vectorizer = TfidfVectorizer()
    tfidf_scores = vectorizer.fit_transform(document_texts)
    vocabulary = vectorizer.get_feature_names_out()
    print("Berechne TF-IDF Gewichte für alle Dokumente und erstelle Elasticsearch-Dokumente ...")
    documents = []
    for document_idx in range(len(document_texts)):
        document_scores = tfidf_scores[document_idx, :].toarray()[0]
        top_k_indices = document_scores.argsort()[-NUM_KEYWORDS_PER_DOCUMENT:][::-1]
        document_keywords = [vocabulary[vocab_idx] for vocab_idx in top_k_indices]

        document = Document(content=document_texts[document_idx],
                            content_type="text",
                            meta={
                                "file_path": document_file_paths[document_idx],
                                "keywords": document_keywords
                            })
        documents.append(document)

    # Die Dokument-Objekte werden jetzt nach Elasticsearch geschrieben.
    print("Schreibe Dokumente nach Elasticsearch ...")
    document_store.delete_documents()  # ... um sicher zu gehen, dass die Dokumente nicht mehrfach geschrieben werden
    document_store.write_documents(documents)

    # Nun werden die Dokumente in Passagen unterteilt.
    passages = []
    for document in document_store.get_all_documents_generator():
        # Zunächst wird jeder Text mithilfe des Tokenizers in Einzelteile zerlegt
        text = document.content
        encodings = tokenizer(text)

        # Die Token-Liste enthält Spezialtokens, die zunächst entfernt werden müssen
        offsets = encodings.encodings[0].offsets
        ids = encodings.encodings[0].ids
        offsets = [offsets[idx] for idx in range(len(offsets)) if ids[idx] not in tokenizer.all_special_ids]

        # Die zuvor via TF-IDF ermittelten Schlüsselworte werden zunächst als Text-Anhang formuliert:
        document_keywords = document.meta['keywords']
        keywords_appendix = "\n\nSchlagworte: " + ";".join(document_keywords)

        # Jetzt kann das Dokument in einzelne Passagen unterteilt werden
        num_tokens = len(offsets)
        passage_cursor = 0
        while passage_cursor < num_tokens:
            passage_offsets = offsets[passage_cursor:passage_cursor + PASSAGE_LENGTH_IN_TOKENS]
            passage_cursor += PASSAGE_LENGTH_IN_TOKENS

            # Wenn eine Passage zu klein ist, dann wird sie verworfen
            if len(passage_offsets) < MIN_PASSAGE_LENGTH_IN_TOKENS:
                continue

            # Ansonsten wird der Text anhand der Tokens rekonstruiert und nach Elasticsearch geschrieben
            start = passage_offsets[0][0]
            end = passage_offsets[-1][1]
            passage_text = text[start:end] + keywords_appendix  # jede Passage wird durch Schlüsselworte ergänzt

            passage = Document(content=passage_text,
                               content_type="text",
                               meta={
                                   "document_id": document.id,  # hiermit wird das Ursprungsdokument wiedergefunden
                                   "document_file_path": document.meta['file_path']
                               })
            passages.append(passage)

    # Auch die Passagen-Objekte werden jetzt nach Elasticsearch geschrieben.
    print("Schreibe Passagen nach Elasticsearch ...")
    passage_store.delete_documents()  # ... um sicher zu gehen, dass die Passagen nicht mehrfach geschrieben werden
    passage_store.write_documents(passages)

    # Mithilfe der Passagen kann nun ein huggingface dataset erstellt werden, auf Basis dessen trainiert werden kann
    print("Erstelle huggingface dataset ...")
    passage_store_credentials = dict(host=ELASTIC_HOST,
                                     username=ELASTIC_USERNAME,
                                     password=ELASTIC_PASSWORD,
                                     index=ELASTIC_PASSAGE_INDEX)
    dataset = datasets.load_dataset(PATH_TO_SCRIPT, passage_store_credentials=passage_store_credentials)
    dataset.save_to_disk(PATH_TO_DATASET)

    print("Fertig.")


if __name__ == '__main__':
    main()
