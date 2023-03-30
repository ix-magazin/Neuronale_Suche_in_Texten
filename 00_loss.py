import torch
from torch import Tensor as T
import torch.nn.functional as F


def sim(q: T, p: T) -> T:
    """
    Berechnet die paarweisen Ähnlichkeiten zwischen einem Tensor aus Anfragevektoren q und einem Tensor aus
    Dokumentvektoren p anhand des Skalarprodukts. Dies kann effizient als Matrixmultiplikation (matmul) implementiert
    werden.

    :param q: Tensor bestehend aus Anfragevektoren der Form (#Anfragen, #Embedding-Dimensionen)
    :param p: Tensor bestehend aus Dokumentvektoren der Form (#Dokumente, #Embedding-Dimensionen.
    :return: Tensor bestehend aus den paarweisen Skalarprodukt-Ähnlichkeiten der Form (#Anfragen, #Dokumente).
    """
    similarity_scores = torch.matmul(q, torch.transpose(p, 0, 1))
    return similarity_scores


def dpr_loss(
        query_vectors: T,
        document_vectors: T,
        positive_idx_per_query: T
) -> T:
    """
    Berechnet den Dense Passage Retrieval Loss gemäß [Karpukhin 2020](https://arxiv.org/abs/2004.04906).

    :param query_vectors: Dies sind die Vektoren zu den Anfragen. Der Tensor hat die Form (#Anfragen, #Embedding-Dimensionen).
    :param document_vectors: Dies sind die Dokumentvektoren, sowohl von den Soll-Dokumenten für jede Anfrage als auch von den zusätzlichen hard negatives. Der Tensor hat die Form (#Dokumente, #Embedding-Dimensionen).
    :param positive_idx_per_query: Dies ist eine Liste an Indizes, die jedem Anfragevektor den Index zum jeweiligen Soll-Dokumentvektor in document_vectors zuordnet.
    :return: Tensor mit den Negative Log Likelihood Loss Werten
    """

    # Folgendes ergibt einen Tensor der Form (#Anfragen, #Dokumente) mit den paarweisen Skalarprodukten.
    # Das heißt, für jede Anfrage q im Minibatch wird der Score zu allen Dokumenten im Minibatch berechnet.
    scores = sim(query_vectors,
                 document_vectors)

    # Es folgt die Loss-Berechnung in zwei Schritten. Dies ist mathematisch äquivalent zu der Loss-Formulierug
    # aus dem Paper und wie sie im iX-Artikel beschrieben steht.

    # 1. Dieser Aufruf führt den Softmax und dessen Logarithmierung in einem Schritt aus.
    #    Beide Operationen separat zu berechnen wäre numerisch instabil, daher gibt es in PyTorch eine
    #    entsprechende optimierte Funktion.
    log_probabilities = F.log_softmax(scores, dim=1)

    # 2. Dieser Aufruf gleicht die log_probabilities mit den Soll-Indizes ab. Das heißt: Für jede Anfrage q soll
    #    das in den Trainingsdaten zugeordnete Ergebnis-Dokument p den höchsten Score verglichen mit allen anderen
    #    Dokumenten haben.
    loss = F.nll_loss(
        log_probabilities,
        positive_idx_per_query,
        reduction="mean",
    )

    # Übrigens: man könnte den Loss auch äquivalent mit folgendem 1-Zeiler berechnen:
    # loss = F.cross_entropy(scores, positive_idx_per_query)
    # Damit wird klar, dass DPR eigentlich fast das Gleiche macht wie CLIP (nur das Ähnlichkeitsmaß unterscheidet sich
    # und einer der Encoder encodiert bei CLIP Bilder statt Text).
    # Dem geneigten Leser bleibt überlassen, sich selbst von der Korrektheit dieser Aussage zu überzeugen.

    return loss
