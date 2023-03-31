import datasets
from datasets import DatasetInfo, DownloadManager
from haystack.document_stores import ElasticsearchDocumentStore

_DESCRIPTION = "Eigener Datensatz für die die Schritt für Schritt Anleitung 'DPR Explained'."

_URL = "https://github.com/schreon/DPR-Explained"


class PassagesDatasetConfig(datasets.BuilderConfig):
    def __init__(self, passage_store_credentials) -> None:
        super().__init__()
        self.passage_store_credentials = passage_store_credentials


class PassagesDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PassagesDatasetConfig

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "text": datasets.Value("string")
            }),
            supervised_keys=None,
            homepage=_URL,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"passage_store_credentials": self.config.passage_store_credentials})]

    def _generate_examples(self, passage_store_credentials):
        passage_store = ElasticsearchDocumentStore(**passage_store_credentials)
        _id = 0
        for passage in passage_store.get_all_documents_generator():
            yield _id, {"text": passage.content}
            _id += 1
