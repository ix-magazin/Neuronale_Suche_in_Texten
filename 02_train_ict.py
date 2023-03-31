import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

from dprexplained.loss import dpr_loss

# Hier bitte den Pfad eintragen, unter welchem in `01_prepare_data.py` der Trainings-Datensatz abgespeichert wurde.
# PATH_TO_DATASET = "/pfad/zu/datensatz"
PATH_TO_DATASET = "/home/ma/s/schroederl/DPR-Explained/dataset"

BASE_PASSAGE_ENCODER = "deepset/gbert-base-germandpr-ctx_encoder"
BASE_QUERY_ENCODER = "deepset/gbert-base-germandpr-question_encoder"
EMBED_SIZE = 768  # muss zu der Embedding-Größe der geladenen vortrainierten Encoder passen

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_BASE_PATH = os.path.join(DIR_PATH, "models", "DPR-Explained-Demo")
OUTPUT_QUERIES_ENCODER_PATH = MODEL_BASE_PATH + "_queries_encoder"
OUTPUT_PASSAGES_ENCODER_PATH = MODEL_BASE_PATH + "_passages_encoder"

UPLOAD_TO_WANDB = False  # Wer Weights&Biases eingerichtet hat, kann hier auf True umschalten.
WANDB_PROJECT_NAME = "DPR-Explained-Demo"  # Ist nur relevant, wenn UPLOAD_TO_WANDB == True.
WANDB_REPORT_STEP = 10  # Ist nur relevant, wenn UPLOAD_TO_WANDB == True.
LOSS_HISTORY_SIZE = 50  # Je größer, desto glatter die dargestellte Loss-Kurve in Weights&Biases

SEED = 42  # Basis-Seed für Zufallszahlen.
BATCH_SIZE = 4  # Wer viel VRAM auf seinen GPUs hat, kann die batch_size hochschrauben. Ansonsten verkleinern.
GRADIENT_ACCUMULATION_STEPS = 1  # Wenn man eine sehr kleine batch_size hat, wi
NUM_EPOCHS = 10  # Für so viele Epochen soll trainiert werden.
NUM_STEPS_PER_EPOCH = 100  # Aus so vielen Epochen soll eine Epoche bestehen.


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()


class ICTCollator:
    """
    Die hier implementierte Form des Inverse Cloze Task Verfahrens weicht von der Formulierung aus dem ursprünglichen
    Paper ab. Anstatt ganze Sätze aus den Passagen zu extrahieren, werden zufällige Substrings extrahiert.

    Dies macht die Umsetzung deutlich einfacher und hat sich unserer Erfahrung nach in der Praxis bewährt.
    """

    def __init__(self,
                 dataset_path,
                 query_tokenizer_name_or_path,
                 passage_tokenizer_name_or_path,
                 rank,
                 batch_size=4,
                 max_query_sequence_length_in_tokens=60,
                 max_passage_sequence_length_in_tokens=200,
                 ict_min_query_length_in_chars=30,
                 ict_max_query_length_in_chars=100
                 ):
        self.ict_min_query_length = ict_min_query_length_in_chars
        self.ict_max_query_length = ict_max_query_length_in_chars
        self.max_query_sequence_length = max_query_sequence_length_in_tokens
        self.max_passage_sequence_length = max_passage_sequence_length_in_tokens
        self.batch_size = batch_size
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_tokenizer_name_or_path)
        self.passage_tokenizer = AutoTokenizer.from_pretrained(passage_tokenizer_name_or_path)
        self.rank = rank
        self.rng = np.random.RandomState(seed=SEED + rank)  # Jeder Worker soll einen eigenen Random Seed verwenden.
        self.dataset = datasets.load_from_disk(dataset_path)
        self.passages = []

        # An dieser Stelle werden alle Passagen in den Speicher geladen. Wer das skalieren möchte, muss hier
        # auf andere Datenstrukturen zurückgreifen.
        for passage_item in self.dataset['train']:
            passage_text = passage_item['text']
            if passage_text not in self.passages:
                self.passages.append(passage_text)

        self.sample_indices = list(range(len(self.passages)))

    def __call__(self, iteration_number):
        batch_queries = []
        batch_passages = []
        batch_positive_idx_per_query = []
        positive_idx = 0
        picked_samples = self.rng.choice(self.sample_indices, self.batch_size, replace=False)
        for sample_index in picked_samples:
            passage_text = self.passages[sample_index]
            text_length = len(passage_text)
            ict_query_length = np.random.randint(self.ict_min_query_length, self.ict_max_query_length)
            ict_query_start = np.random.randint(0, text_length - ict_query_length)
            ict_query_text = passage_text[ict_query_start:ict_query_start + ict_query_length]
            batch_queries.append(ict_query_text)
            batch_passages.append(passage_text)
            batch_positive_idx_per_query.append(positive_idx)
            positive_idx += 1

        batch_queries = self.query_tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_queries,
                                                               padding="max_length",
                                                               truncation=True,
                                                               max_length=self.max_query_sequence_length,
                                                               return_tensors="pt")

        batch_passages = self.passage_tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_passages,
                                                                  padding="max_length",
                                                                  truncation=True,
                                                                  max_length=self.max_passage_sequence_length,
                                                                  return_tensors="pt")

        batch_positive_idx_per_query = torch.tensor(batch_positive_idx_per_query)

        return batch_queries, batch_passages, batch_positive_idx_per_query


def main(rank, world_size):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    try:
        collator = ICTCollator(
            PATH_TO_DATASET,
            BASE_QUERY_ENCODER,
            BASE_PASSAGE_ENCODER,
            rank,
            batch_size=BATCH_SIZE,
            max_query_sequence_length_in_tokens=60,
            max_passage_sequence_length_in_tokens=200,
            ict_min_query_length_in_chars=30,
            ict_max_query_length_in_chars=100)

        num_total_iterations = NUM_EPOCHS * NUM_STEPS_PER_EPOCH

        loader = DataLoader(TensorDataset(torch.range(1, num_total_iterations, dtype=torch.int)),
                            batch_size=1,
                            num_workers=1,
                            prefetch_factor=2,
                            pin_memory=True,
                            collate_fn=collator)

        print("loading pretrained models ...")
        passages_encoder = DPRContextEncoder.from_pretrained(BASE_PASSAGE_ENCODER)
        queries_encoder = DPRQuestionEncoder.from_pretrained(BASE_QUERY_ENCODER)

        passages_encoder.to(rank)
        passages_encoder = DDP(passages_encoder, device_ids=[rank], broadcast_buffers=False)

        queries_encoder.to(rank)
        queries_encoder = DDP(queries_encoder, device_ids=[rank], broadcast_buffers=False)

        optimizer = Adam(list(passages_encoder.parameters()) + list(queries_encoder.parameters()),
                         lr=1e-6,
                         betas=(0.9, 0.95),
                         eps=1e-8)

        def get_embeddings(encodings, encoder):
            embeds = encoder(input_ids=encodings['input_ids'].to(rank),
                             attention_mask=encodings['attention_mask'].to(rank))

            return embeds.pooler_output

        if rank == 0:
            data_iterator = iter(tqdm(loader))
        else:
            data_iterator = iter(loader)

        torch.autograd.set_detect_anomaly(True)
        wandb_initialized = False
        # set training flags
        queries_encoder.train()
        passages_encoder.train()
        print("%d iterations per epoch" % NUM_STEPS_PER_EPOCH)
        loss_history = []
        for epoch in range(NUM_EPOCHS):
            for it in range(NUM_STEPS_PER_EPOCH):
                batch_queries, batch_passages, batch_positive_idx_per_query = next(data_iterator)
                batch_positive_idx_per_query = batch_positive_idx_per_query.to(rank)

                queries_embeds = get_embeddings(batch_queries, queries_encoder)
                passages_embeds = get_embeddings(batch_passages, passages_encoder)

                queries_embeds = queries_embeds.reshape((BATCH_SIZE, EMBED_SIZE))
                passages_embeds = passages_embeds.reshape((BATCH_SIZE, EMBED_SIZE))

                loss = dpr_loss(queries_embeds, passages_embeds, batch_positive_idx_per_query)

                loss.backward()

                if it % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if rank == 0:
                    loss_history.append(loss.item())
                    if len(loss_history) >= LOSS_HISTORY_SIZE:
                        loss_history = loss_history[-LOSS_HISTORY_SIZE:]

                        if it % WANDB_REPORT_STEP == 0:
                            if UPLOAD_TO_WANDB and not wandb_initialized:
                                print("init wandb")
                                # start a new wandb run to track this script
                                wandb.init(
                                    # set the wandb project where this run will be logged
                                    project=WANDB_PROJECT_NAME
                                )
                                wandb_initialized = True

                            if UPLOAD_TO_WANDB:
                                running_loss = np.sum(loss_history) / (BATCH_SIZE * LOSS_HISTORY_SIZE)
                                wandb.log(
                                    {"running_loss": running_loss,
                                     "epoch": epoch})

            if rank == 0:
                queries_encoder.module.save_pretrained(OUTPUT_QUERIES_ENCODER_PATH)
                passages_encoder.module.save_pretrained(OUTPUT_PASSAGES_ENCODER_PATH)

                # Zusätzlich müssen Kopien der Tokenizer gespeichert werden
                query_tokenizer = AutoTokenizer.from_pretrained(BASE_QUERY_ENCODER)
                query_tokenizer.save_pretrained(OUTPUT_QUERIES_ENCODER_PATH)
                passage_tokenizer = AutoTokenizer.from_pretrained(BASE_PASSAGE_ENCODER)
                passage_tokenizer.save_pretrained(OUTPUT_PASSAGES_ENCODER_PATH)

        #################
        if rank == 0:
            if UPLOAD_TO_WANDB:
                wandb.finish()
    finally:
        if rank == 0:
            print("cleaning up ...")
        cleanup()

    if rank == 0:
        print("finished.")


def run_parallel(function_to_run, world_size):
    mp.spawn(function_to_run,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    num_gpus = 1  # wer mehr GPUs hat als eine, kann dies hier eintragen, um sie alle parallel zu nutzen.
    run_parallel(main, num_gpus)
