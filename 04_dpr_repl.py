import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline

# In der Form bitte nur für einen lokalen Development-Server nutzen. Auf keinen Fall Credentials in Produktivcode
# schreiben.
ELASTIC_HOST = "localhost"
ELASTIC_USERNAME = ""
ELASTIC_PASSWORD = ""
ELASTIC_PASSAGE_INDEX = "demo_passages"

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_BASE_PATH = os.path.join(DIR_PATH, "models", "DPR-Explained-Demo")
QUERIES_ENCODER_PATH = MODEL_BASE_PATH + "_queries_encoder"
PASSAGES_ENCODER_PATH = MODEL_BASE_PATH + "_passages_encoder"


def main():
    passage_store = ElasticsearchDocumentStore(host=ELASTIC_HOST,
                                               username=ELASTIC_USERNAME,
                                               password=ELASTIC_PASSWORD,
                                               index=ELASTIC_PASSAGE_INDEX)

    retriever = DensePassageRetriever(passage_store,
                                      query_embedding_model=QUERIES_ENCODER_PATH,
                                      passage_embedding_model=PASSAGES_ENCODER_PATH,
                                      embed_title=False,
                                      use_fast_tokenizers=True, )

    p_retrieval = DocumentSearchPipeline(retriever)

    running = True

    while running:
        query = input("Suchanfrage: ")

        print("Suchergebnisse: ")
        print("--------------------------")
        res = p_retrieval.run(query=query,
                              params={"Retriever": {"top_k": 3}})

        for passage in res['documents'][::-1]:
            print(passage.content)
            print("-- Dateipfad: %s" % passage.meta['document_file_path'])
            print("--------------------------")


if __name__ == '__main__':
    main()
