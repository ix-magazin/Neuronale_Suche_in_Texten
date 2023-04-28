# Dense Passage Retrieval für die eigene Domäne

Wer mehr über neuronale Suche erfahren will, muss aktuell einige fragmentierte Quellen und
wissenschaftliche Paper in englischer Sprache wälzen. Dense Passage Retrieval (DPR) ist ein neuronales Suchverfahren,
das mithilfe eines Frage-Texts Text-Dokumente durchsuchen kann. Dieser Artikel
soll einen sanften Einstieg in die (gar nicht so harte) Mathe liefern, die dahinter steckt.
Grundlegende Technik wie Tokenization,die Transformer-Architektur und Attention-Mechanismen beschreibt dieser Artikel
nicht - diese Informationen sind im Internet aber leicht auffindbar (siehe **LINK**).

Zusätzlich zu diesem Artikel zeigt ein Begleit-Repository auf GitHub, wie sich diese Mathe in Python Code implementieren
lässt (siehe **LINK**). Ein Folge-Artikel im kommenden iX Sonderheft zeigt, wie man mit dem resultierenden Modell in Kombination mit dem
Framework haystack ein produktionstaugliches neuronales Suchsystem aufbauen kann. Zusätzlich wird dann ein Datensatz veröffentlicht,
der zusammen mit der Landeshauptstadt München entwickelt wurde. Wer also über keinen eigene Dokumentsammlung verfügt,
kann auf das Sonderheft warten und dann loslegen.

## TF•IDF

Beim Verständnis von DPR hilft zunächst der Vergleich zu den weit verbreiteten Information Retrieval Systemen, die auf
TF•IDF basieren. Da TF•IDF als gute Baseline dient und auch in der Ära der neuronalen Netze seine Daseinsberechtigung
hat, folgt hier ein kurzer Umriss dieser Methode. Der Artikel erklärt nur die wesentlichen Punkte, die für den
Vergleich mit DPR hilfreich sind.

TF•IDF steht für *Term Frequency • Inverse Document Frequency*, zu Deutsch etwa *Termhäufigkeit • Inverse
Dokumenthäufigkeit*. Der Name verrät schon, dass es bei diesem statistischen Maß im wesentlichen darum geht, Terme (also
Wörter) in Dokumenten zu zählen. Dabei ist das Ziel, insbesondere Wörter zu zählen, die ein
Dokument von allen anderen Dokumenten besonders unterscheidbar machen . Genauer ausgedrückt möchte man eine Funktion, die
einen gegebenen Term $t$ und ein Dokument $d$ betrachtet und besonders hohe Werte liefert, wenn $t$ häufig in dem
Dokument vorkommt und gleichzeitig sehr spezifisch ist. Die Werte sind niedrig, wenn $t$ in dem Dokument nur selten
vorkommt oder ein sehr geläufiger, unspezifischer Term ist. Dies erreicht man durch das Produkt der Termhäufigkeit (TF)
mit dem Logarithmus der inversen Dokumenthäufigkeit (IDF), resultierend in TF•IDF.

Die Termhäufigkeit $\mathrm{tf}$ sagt aus, wie oft ein Term $t$ in einem Dokument vorkommt, relativ zur Länge des
Dokuments. Die Dokumentenlänge ist gleich der Summe der Häufigkeiten aller bekannten Terme $t$ im Dokument $d$ ist). Mathematisch ist die Termhäufigkeit wie folgt definiert:

$\text{tf}(t, d) = \frac{\text{Haeufigkeit des Terms t in Dokument d}}{\text{Anzahl aller Woerter in Dokument d}} = \frac{f _ {t,d}}{{\sum _ {t' \in d}{f _ {t',d}}}}$

Nun gilt es abzubilden, wie spezifisch ein Term ist; das heißt, wie gut er sich als Indikator für das
Unterscheiden von Dokumenten eignet. Dabei gilt die Annahme, dass ein Term dann besonders spezifisch ist,
wenn er nur in wenigen Dokumenten vorkommt. Sogenannte Stoppwörter wie *die*, *aber* und *oder* kommen in fast allen
!!!ist oder hier auch als Beipiel möglich? Das Ende der Aufzählung sollte ein und enthalten.
Dokumenten vor und haben somit eine sehr geringe Spezifizität. Ein Begriff wie Urlaubsantrag kommt hingegen nur in
einem Bruchteil aller Dokumente vor und ist somit deutlich spezifischer als ein Stoppwort. Intuitiv beschreibt das
folgende Verhältnis die Dokumenthäufigkeit:

$\frac{\text{Anzahl der Dokumente, die den Term t enthalten}}{\text{Anzahl aller Dokumente}}$

Je höher die Dokumenthäufigkeit, desto geläufiger ist ein Term $t$ in allen vorliegenden Dokumenten $D$. Eine Funktion $\mathrm{idf(t, D)}$ 
soll dann hohe Werte liefern, wenn der Term $t$ nur in wenigen Dokumenten vorkommt, somit also spezifisch ist (so wie Urlaubsantrag).
Im Gegensatz dazu soll die Funktion niedrige Werte liefern, wenn der Term sehr geläufig ist, also in sehr vielen Dokumenten vorkommt (so wie
beispielsweise die Stoppwörter *die*, *aber* und *oder*). Dies erreicht man, indem man den Logarithmus des Inversen des oben
genannten Verhältnis betrachtet. Die Funktion $\mathrm{idf}(t, D)$ ist somit definiert als:

$\mathrm{idf}(t, D) = \log \frac{N}{|d \in D: t \in d|}$

Das zusammengesetzte TF•IDF Maß ist dann das Produkt $\mathrm{tf}(t, d) \cdot \mathrm{idf}(t, D)$. Es hängt von einem
Term $t$, einem Dokument $d$ und einer Dokumentsammlung $D$ ab, mit $t \in d$ und $d \in D$. Das Maß hat dann hohe
Werte, wenn $t$ sowohl oft in $d$ vorkommt, aber selten in der gesamten Dokumentsammlung $D$. Kommt $t$ oft in $d$ vor,
aber auch oft in der gesamten Dokumentsammlung $D$, dann ist der Wert trotzdem niedrig, weil der Logarithmus in der
$\mathrm{idf}$ Funktion das Produkt in diesem Fall stark gegen 0 zieht.

Um zwei Dokumente $d _ 1$ und $d _ 2$ miteinander vergleichen zu können, wird die Statistik der TF•IDF Werte über alle Terme
$t$ aus dem gesamten Vokabular $V$ herangezogen. Dabei erstellt man für jedes Dokument $d$ einen Dokumentvektor $v _ d$,
der sich aus den einzelnen TF•IDF Werten zusammensetzt. Somit repräsentiert jede Dimension $i$ in einem solchen
Dokumentvektor $v _ d$ die Statistik über ein bestimmtes Wort $t _ i$ aus dem Vokabular $V$ im vorliegenden
Dokument $d$:

$v _ {d, i} = \mathrm{tf}(t _ i, d) \cdot \mathrm{idf}(t _ i, D)$

Um die Ähnlichkeit zweier Dokumente $d _ 1$ und $d _ 2$ festzustellen, lässt sich eine Ähnlichkeit zwischen ihren beiden
Dokumentvektoren $v _ {d _ 1}$ und $v _ {d _ 2}$ berechnen. Hierzu ist zunächst ein Ähnlichkeitsmaß notwendig. Üblich im
Information Retrieval ist an der Stelle die Kosinusähnlichkeit, die folgendermaßen definiert ist:

$\mathrm{cosine _ sim}(q, p) = \frac{q \cdot p}{\max(\lVert q {\rVert} _ 2 \cdot \lVert p {\rVert} _ 2, \epsilon)}$

Wenn man nun annimmt,
Nun lässt sich die TF•IDF Statistik einer Suchanfrage mit der TF•IDF Statistik eines Dokuments vergleichen.
Dafür kann man annehmen, das eine hohes Übereinstimmen der Werte für eine semantische Übereinstimmung steht und damit eine 
k-nearest-neighbor-Suche bauen. Wie stark oder schwach diese Annahme in der Praxis greift, spürt man, wenn man in der
Intranetsuche seiner Organisation beispielsweise nach "Urlaubsantrag" sucht,aber keinen einzigen Treffer erzielt,
weil in den relevanten Dokumenten nur von "Antrag auf Urlaub" die Rede ist. Das liegt daran, dass diese Modellierung
keine Repräsentierung von Sprache an sich beinhaltet und über kein Weltmodell verfügt, sondern ausschließlich die
Häufigkeiten bestimmter Buchstabenfolgen berücksichtigt. Somit besteht zwischen dem Wort $t _ 1 = \text{"Urlaubsantrag"}$
und $t _ 2 = \text{"Urlaub"}$ zwar ein semantischer Zusammenhang, die TF•IDF Modellierung berücksichtigt diese Semantik aber in
keiner Weise. Somit hat Urlaubsantrag nichts mit Urlaub zu tun. Auch der Plural Urlaubsanträge hat
keinen Zusammenhang dem Singular Urlaubsantrag.

In der Praxis versucht man, diese Schwächen von TF•IDF mit verschiedenen Maßnahmen zu dämpfen: Beispielsweise normalisiert man die
Texte, indem man Wörter alle in Kleinschreibweise und Stammformen überführt. Zusätzlich entfernt man
Umlaute, Akzente und so weiter. Auch das Abbilden auf andere Terme mithilfe von Synonymlisten ist gängig. Auf die Weise
wird dann aus "Urlaubsantrag" etwa "urlaubsantrag" und aus "Urlaubsanträge" ebenfalls "urlaubsantrag". Dann kann
man nach Urlaubsanträge suchen und findet trotzdem Dokumente, in denen lediglich Urlaubsantrag vorkommt, weil das System beides
auf die Stammform "urlaubsantrag" abbildet.

Es sollte aber klar sein, dass auf TF•IDF basierende Information Retrieval Systeme an Grenzen stoßen,
die in fundamentalen Mängeln begründet liegen: Es fehlt ein semantischer Zusammenhang zwischen den Termen,
sie tragen keinerlei Kontextinformation, die Bedeutung des Satzgefüges und die Feinheiten der verwendeten
Sprache nicht berücksichtigt TF•IDF nicht.

## Von TF•IDF zu Sprachmodellen und DPR

Sprachmodelle haben das Potenzial, die Grenzen von wortfrequenzbasierten Verfahren wie TF•IDF zu überwinden. Dies
gelingt insbesondere mit tiefen neuronalen Netzen vom Typ Transformer, die auf
dem Attention-Mechanismus beruhen. Das in diesem Artikel vorgestellte Verfahren Dense Passage Retrieval (DPR)
basiert auf solchen Sprachmodellen und ist eine Alternative zu TF•IDF Retrieval Systemen.

![DPR verglichen mit TFIDF](DPR_vgl_TFIDF.png)
!!!Bitte noch eine Bildunterschrift hinzufügen.

Wie bei TF•IDF bildet DPR sowohl die Anfragen als auch die suchbaren Dokumente in denselben hochdimensionalen Raum
mit $D$ Dimensionen ab. Innerhalb von diesem kann man eine k-nearest-neighbor-Suche durchgeführen.
Einer der wesentlichen Unterschiede von DPR gegenüber TF•IDF besteht darin, dass die $D$ Dimensionen nicht für
die Frequenz des Vorkommens eines bestimmten Schlüsselworts stehen. Beim DPR-Verfahren definieren die $D$ Dimensionen die
Koordinaten in einem $D$-dimensionalen Raum. Sie sind somit wie die `x`, `y` und `z` Koordinaten im 3D-Raum zu
interpretieren. Wer möchte, kann versuchen, sich ein Koordinatensystem mit 768 Koordinaten-Achsen vorzustellen. Die
Aufgabe der neuronalen Netze ist, diese Punkte so anzuordnen, dass zueinander semantisch ähnliche Dokumente in diesem
Raum nah beieinander liegen. Was in diesem Zusammenhang "Nähe" bedeutet, bestimmt das gewählte Ähnlichkeitsmaß.
Weiterhin sind bei TF•IDF die meisten Dimensionen in den Anfrage- und Dokumentenvektoren gleich null, weil
die meisten Schlüsselworte aus dem Vokabular in einem einzelnen Text nicht vorkommen. Es handelt sich somit bei TF•IDF
um einen dünnbesetzten Vektor (sparse vector). Bei DPR ist jede Dimension der Anfrage- und Dokumentenvektoren
meistens ungleich null, da die x/y/z-Koordinaten der Objekte in einem 3D-Raum auch meistens nicht 0 sind, sondern
irgendwo im Raum verteilt liegen. Dieser Unterschied erklärt auch das Wort dense (engl.: dicht) im Namen des
Verfahrens Dense Passage Retrieval. Ein weiterer Unterschied von DPR verglichen mit Systemen basierend auf TF•IDF
liegt im verwendeten Ähnlichkeitsmaß, anhand von dem man die k-nearest-neighbor-Suche durchführt. TF•IDF beruht
üblicherweise auf der oben beschriebenen Kosinusähnlichkeit. Von den Autoren des DPR-Papers wurde jedoch gezeigt, dass
sich für DPR das bloße Skalarprodukt als Ähnlichkeitsmaß besser eignet. Das Skalarprodukt bleibt übrig, wenn man in der
oben beschriebenen Formulierung der Kosinusähnlichkeit die Normalisierung streicht. Dies ergibt ein Ähnlichkeitsmaß,
welches im Folgenden $\text{sim}$ genannt wird:

$\text{sim}(q, p) = q \cdot p$

Nun stellt sich die Frage, wie die das DPR-Verfahren die Anfragevektoren $q$ und die Dokumentvektoren $p$ berechnet.
Der in diesem Artikel vorgeschlagene Lösungsansatz ist eine von vielen Möglichkeiten, wie man ein
DPR-Modell implementieren kann. Die Implementierung im Begleit-Repository dieses Artikels orientiert sich
stark an der von den Autoren des originalen DPR-Papers vorgeschlagenen Variante und der dazugehörigen Implementierung (siehe **LINK**).
Sie ist mit der DPR-Implementierung im huggingface `transformers`-Paket kompatibel (ebenfalls unter **LINK**). 
Das DPR-Verfahren ist grundsätzlich mit allen Typen von neuronalen Netzen umsetzbar, die in der Lage sind,
Text einzulesen und Repräsentationsvektoren für den eingelesenen Text zu generieren. Als Ausgangspunkt sind
aber nach dem aktuellen Stand der Technik vor allem vortrainierte Encoder-Sprachmodelle sinnvoll.

Ein solches Encoder-Sprachmodell liefert für eine Eingabe-Sequenz aus Tokens eine korrespondierende Ausgabe-Sequenz
derselben Länge, bestehend aus kontextualisierten Embedding-Vektoren. Der Encoder stellt dem tokenisierten Text ein
spezielles `[CLS]`-Token voran, das dem Klassifizieren des Textes dient. Man extrahiert die Aktivierungen der letzten Schicht aus dem Transformer-Netz an der Stelle des `[CLS]`-Tokens und verwendet sie als Repräsentationsvektor `q` (Anfragen) oder `p` (Dokumente).

Prinzipiell ließe sich dasselbe Sprachmodell sowohl für das Berechnen der Anfragevektoren `q` als auch für die
Dokumentenvektoren `p` verwenden. Die Autoren des DPR-Papers empfehlen aber zwei Instanzen desselben
ursprünglichen Sprachmodells: ein Encoder für die Anfragen, im huggingface `transformers`-Paket eingebettet in die
Klasse `DPRQuestionEncoder`, und ein Encoder für die Dokumente, analog eingebettet in die Klasse `DPRContextEncoder`.

## Loss-Funktion bei Dense Passage Retrieval

Wenn man eine Anzahl $Q$ Anfragevektoren $q _ i$ der Länge $D$ zusammenfasst, erhält man einen Tensor $q$ des Rangs 2 und
!!!Tensor des Rangs 2 könnte nicht allen Lesern klar sein, könnt ihr das noch kurz erklären?
der Form $(Q, D)$. Analog lassen sich die $P$ Dokumentenvektoren $p _ j$ zu einem Tensor $p$ des Rangs 2 mit der Form $(P, D)$
zusammenfassen. Dann kann man die paarweisen Ähnlichkeiten $\text{sim}(q _ i, p _ j)$ der Anfragen und Dokumente
mithilfe einer Matrixmultiplikation von $q$ und $p$ berechnen. Der resultierende Tensor hat dann ebenfalls den Rang 2
und die Form $(Q, P)$. Die Funktion `sim` zeigt Listing 1. Sie ist im begleitenden Git-Repository in `dprexplained/loss.py` zu finden.

::: listing
# Listing 1: sim-Funktion für das Ähnlichkeitsmaß
``` python
import torch
from torch import Tensor as T
import torch.nn.functional as F


@torch.jit.script
def sim(q: T, p: T) -> T:
    similarity_scores = torch.matmul(q, torch.transpose(p, 0, 1))
    return similarity_scores
```    
:::

Dieses Ranking-Problem lässt sich als Multi-Klassifikationsproblem betrachten.
Gegeben seien der Anfragevektor $q _ i$ mit dem Soll-Dokumentenvektor $p _ i^+$ und eine Menge an irrelevanten
Dokumentenvektoren $p _ {i,j}^-$. Die Menge aller gegebenen Dokumentenvektoren ist dann $P _ i = \{p _ i^+, p _ {i, 1}^-, \dots, p _ {i, n}^- \}$. Nun lässt sich eine Wahrscheinlichkeit formulieren, die in etwa folgendes aussagt: Wie wahrscheinlich
ist es, dass ein Klassifikator in Anbetracht einer Anfrage $q _ i$ den korrekten Soll-Dokumentenvektor $p _ i^+$ aus der
Menge aller Dokumente $P _ i$ auswählt? Für jeden Dokumentenvektor $p$ gibt es dann eine entsprechende Klasse. Wichtig:
Hier dient die Modellierung als Wahrscheinlichkeit nur als Vehikel, um einen geeigneten Loss zu formulieren und das
Modell zu trainieren. Im Produktionsbetrieb beziehungsweise zur Inferenzzeit ist das Verfahren deterministisch und nicht
stochastisch.

Zum Modellieren einer Wahrscheinlichkeitsverteilung über mehrere Klassen eignet sich die *Softmax-Funktion*. Die wandelt einen
Vektor $z$ aus $|K|$ reellen Zahlen in eine Wahrscheinlichkeitsverteilung um. Die reellen Zahlen repräsentieren dabei
jeweils einen unnormalisierten Score $z _ i$ für die Wahrscheinlichkeit einer Klasse $k _ i \in K$.
Die Wahrscheinlichkeit $\sigma(z) _ i$, dass die Funktion die Klasse $k _ i$ aus der Menge möglicher Klassen $K$ auswählt, berechnet sich gemäß der
Softmax-Formulierung wie folgt:

<!-- $\sigma(z) _ i = \frac{e^{z _ i}}{\sum _ {j=1}^{\left|K\right|} e^{z _ j}}$ --> <img style="transform: translateY(0.1em); background: white; height: 50px;" src="svg\u20xjHJjAM.png">

Betrachtet man die Ausgabe der oben definierten Ähnlichkeitsfunktion $\text{sim}$ als den unnormalisierten Score $z _ i$,
so kann man diese in die Softmax-Formulierung einsetzen:

<!-- $\sigma(z) _ i = \frac{e^{\text{sim}(q _ i,p _ {i}^+)}}{e^{\text{sim}(q _ i,p _ {i}^+)} + \sum _ {j=1}^n{e^{\text{sim}(q _ i,p _ {i,j}^-)}}}$ --> <img style="transform: translateY(0.1em); background: white; height: 50px;" src="svg\WrSYgkwGCR.png">

Die übliche Methode, eine Wahrscheinlichkeit zu maximieren, ist, die Negative Log Likelihood (NLL) zu minimieren. Und
damit gelangt man zur Formulierung des Loss $L$:

<!-- $L(q _ i, p{ _ i}^+, p _ {i, 1}^-, ..., p _ {i, n}^-)=-\log(\sigma(z) _ i)=-\log\frac{e^{\text{sim}(q _ i,p _ {i}^+)}}{e^{\text{sim}(q _ i,p _ {i}^+)} + \sum _ {j=1}^n{e^{\text{sim}(q _ i,p _ {i,j}^-)}}}$ --> <img style="transform: translateY(0.1em); background: white; height: 50px;" src="svg\tPImr85tkf.png">
!!!Ich habe überlegt, diesen Teil ab Zeile 167 in einen Extrakasten auszulagern. Lässt sich das nahtlos verschieben?

Am besten wäre es, für jedes Anfrage/Dokument-Paar $(q _ i, p{ _ i}^+)$ alle anderen Dokumente aus dem vorliegenden
Datensatz als Negativbeispiele $p _ {i,j}^-$ zu verwenden. Dies ist aber mit Hinblick auf den Rechen- und Speicheraufwand
nicht machbar. Daher gibt es in der Praxis einen Trick, bei dem man nur ein Bruchteil der Dokumente zur selben Zeit
lädt und zum Berechnen des Loss heranzieht. Dabei wählt man gleich mehrere Anfrage/Dokument-Paare
$(q _ i, p{ _ i}^+)$ aus dem Gesamtdatensatz aus, die einen Minibatch formen. Nun werden für jede Anfrage
$q _ i$ aus diesem Minibatch alle anderen Dokumente $p{ _ j}^+$ als Negativdokumente für $q _ i$ betrachtet. Diese bezeichnet
man als in-batch negatives. Der Liste an Negativdokumenten lassen sich zusätzlich noch hard negatives
hinzufügen. Damit sind zusätzliche Negativdokumente gemeint, die den eigentlichen Soll-Dokumenten pro Anfrage
sehr ähnlich sind und sich somit schwer unterscheiden lassen. Eine mögliche Implementierung des oben hergeleiteten Loss in Kombination mit dem in-batch negatives-Trick zeigt Listing 2.

::: listing
# Listing 2: Herleiten des Loss mit in-batch negatives
``` python
@torch.jit.script
def dpr_loss(
        query_vectors: T,
        document_vectors: T,
        positive_idx_per_query: T
) -> T:

    scores = sim(query_vectors,
                 document_vectors)

    log_probabilities = F.log_softmax(scores, dim=1)

    loss = F.nll_loss(
        log_probabilities,
        positive_idx_per_query,
        reduction="mean",
    )

    return loss
```
:::

## Passagen aufteilen

Sprachmodelle sind sehr rechenintensiv. Das liegt vor allem daran, dass der Attention-Mechanismus in der Transformer-Architektur
sowohl in der Rechen- als auch der Speicherkomplexität quadratisch mit der Sequenzlänge wächst.
Das heißt, dass die in akzeptabler Zeit und mit akzeptablen Resourcen berechenbare Textlänge
begrenzt ist. Zwar kann man diesen Resourcenhunger durch verschiedene Maßnahmen dämpfen (etwa durch sogenannte 
Sparse Attention), die vorliegenden Anleitung begrenzt jedoch einfach die maximale Sequenzlänge auf 200 Tokens.
Dies hat zur Folge, dass man die vorliegenden Texte in Passagen einteilen muss. Dies
ist zeigt Listing 3, im GitHub-Repository ist das in `01_prepare_data.py` enthalten (**LINK**).

::: listing
# Listing 3: Einteilen von Dokumenten in Passagen
```python
# Hier muss der DPR Passage Encoder spezifiziert werden (auch Context Encoder genannt). Kann auch ein lokaler Pfad sein.
BASE_DPR_MODEL_NAME_OR_PATH = "deepset/gbert-base-germandpr-ctx_encoder"

# Hier wird spezifiziert, wie groß Passagen sein sollen.
PASSAGE_LENGTH_IN_TOKENS = 200
MIN_PASSAGE_LENGTH_IN_TOKENS = 20

# Hier wird spezifiziert, wie viele Schlüsselworte an die Passagen angehängt werden sollen
NUM_KEYWORDS_PER_DOCUMENT = 10
```
:::

Die einzelnen Passagen verlieren allerdings den Kontext des Gesamtdokuments. Dadurch sind manche Passagen nicht mehr
sinnvoll suchbar. Beispielsweise könnte ein Artikel, der von den verschiedenen Regeln und Zuständigkeiten rund um die
Hundesteuer handelt, am Ende eine Address- und Telefonliste enthalten. Wäre dieser Artikel besonders lang, dann würde
diese Liste womöglich in einer separaten Passage landen. Stellt man nun die Anfrage "Wo muss ich anrufen, wenn ich
meinen neuen Hund zur Hundesteuer anmelden will?", dann kann das Verfahren die entsprechende Antwort nicht finden. Ein
einfacher Trick ist an der Stelle, wiederum TF•IDF zu nutzen. Damit lassen sich für jedes Dokument geeignete Schlüsselwörter
ermitteln. Wird das Dokument dann in einzelne Passagen aufgeteilt,hängt man die Schlüsselworte jeweils an.
Auf diese Weise erhält jede Passage einen Kontext abhängig vom Gesamtdokument.

## Training von Dense Passage Retrieval Modellen

Um ein DPR-Modell zu trainieren, sind Daten bestehend aus Such-Anfragen und zugeordneten Soll-Ergebnisdokumenten
notwendig. Diese sind sehr aufwändig zu erheben, daher zweckentfremdet man in der Praxis bestehende Frage & Antwort Datensätze wie etwa SQuAD.
Leider sind öffentliche und zudem deutschsprachige Datensätze dieser Art rar. Ein solcher Datensatz ist
deepset's GermanDPR, welcher von GermanQUaD abgeleitet wurde. Das dazugehörige Paper findet sich unter **LINK**ist.
Allerdings beziehen sich diese Datensätze auf sehr allgemein gestreutes Wissen über einen Teil der deutschen Wikipedia.
Liegen sehr domänenspezifische Dokumente vor, in denen sich viel Fachvokabular findet, dann liefern mit GermanDPR trainierte Modelle
oftmals unzufriedenstellende Ergebnisse. In solchen Fällen sind oftmals sogar klassische TF•IDF Systeme besser, weil diese nicht erst den
Zusammenhang der Sprache lernen müssen. Um ein allgemeines, neuronales DPR-Modell auf das eigene Fachvokabular zu
spezialisieren, wäre es optimal, einen domänenspezfisichen Frage-und-Antwort-Datensatz zu annotieren. Das ermöglicht
beispielsweise das haystack annotation tool (siehe **LINK**).

Doch selbst wenn keine annotierten Daten zur Verfügung stehen, kann man DPR-Modelle auf eine spezifische Domäne anpassen.
Dies geht beispielsweise über die Inverse Cloze Task Methode (ICT) (siehe **LINK**). Dabei wendet man einen sehr
simplen Hack an: Aus einem Dokument werden zufällige Sätze gewählt, die man dann als Pseudo-Anfrage behandelt. Eine
vereinfachte Implementierung von ICT ist im GitHub-Repository in `02_train_ict.py` umgesetzt. Auf diese
Weise lernt ein bereits vortrainiertes Sprachmodell, mit domänenspezifischen Begriffen und Spracheigenheiten umzugehen.

## Fazit

Mit DPR lassen sich die Schwächen von Suchsystemen, die lediglich auf die Häufigkeit von Schlüsslworten aufbauen,
überwinden. Durch ein existierendes Sprachmodell als Ausgangspunkt wird es möglich, den Sinn einer Suchanfrage zu
erfassen, um ihn beim Ranking der Suchergebnisse zu berücksichtigen. Als Ausgangspunkt eignen sich allgemeine
DPR-Modelle, die zuvor anhand von vortrainierten Encoder-Sprachmodellen und großen Frage/Antwort-Datensätzen erstellt
wurden. Mithilfe von ICT lassen sich die dann auf die eigene Domäne verfeinern.

Wenn man den Aufwand nicht scheut, zusätzliche Trainingsdaten zu annotieren, kann man beispielsweise
das haystack annotation tool verwenden, um einen eigenen, domänenspezifischen Frage-und-Antwort-Datensatz zu erstellen.
Der ist dann gleich für zwei Aspekte nützlich: Erstens kann man damit das DPR-Modell nochmals stärker auf die eigene Domäne
spezialisieren, und zweitens kann man damit dann ein zusätzliches Extractive Question Answering Modell trainieren,
das einem die Antwort-Stellen in den Ergebnisdokumenten markiert. 

Anhand vom Source Code im Begleit-Repository(https://github.com/schreon/DPR-Explained) kann, wer möchte, ein DPR-Suchmodell auf den eigenen
Daten trainieren und ausprobieren (siehe **LINK**). Wer keine eigenen Daten zur Verfügung hat, kann sich auf das kommende iX Sonderheft
freuen: Der darin enthaltenen Folge-Artikel zeigt, wie man mit `haystack` eine produktionstaugliche Extractive Question Answering und DPR Pipeline erstellen kann. Als Grundlage dient dabei ein Datensatz der Landeshauptstadt München.

# Dense Passage Retieval Links

- [GitHub-Repository zum Artikel](https://github.com/schreon/DPR-Explained bzw. dann der Fork)
- [Original-Paper zu DPR](https://arxiv.org/abs/2004.04906)
- [Implementierung zum DPR-Paper](https://github.com/facebookresearch/DPR)

# Grundlegende Technik
- [Tokenization](https://huggingface.co/docs/transformers/main_classes/tokenizer),
- [Transformer-Architektur](https://huggingface.co/docs/transformers/model_summary)
- [Attention-Mechanismen](https://huggingface.co/docs/transformers/main/attention)

# Ressourcen
- [DPR-Implementierung im huggingface `transformers`-Paket](https://huggingface.co/docs/transformers/model_doc/dpr)
- [Paper zu GermanQuAD und GermanDPR](https://arxiv.org/abs/2104.12741)
- [haystack annotation tool](https://docs.haystack.deepset.ai/docs/annotation)
- [Inverse Cloze Task](https://aclanthology.org/P19-1612/)
