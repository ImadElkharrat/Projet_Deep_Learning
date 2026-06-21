"""
================================================================================
PROJET DE FIN DE MODULE – DEEP LEARNING
EMSI Casablanca | Année universitaire 2025–2026
================================================================================
PARTIE III – RNN, LSTM, GRU ET SEQ2SEQ
Tâches :
  • Modélisation de langage (prédiction du token suivant) – corpus intégré
  • Traduction de séquences (Seq2Seq) – corpus chiffres ↔ lettres (synthétique)
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math, time, copy, re, warnings
from collections import Counter
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import FancyBboxPatch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("PARTIE III – RNN / LSTM / GRU / SEQ2SEQ")
print("=" * 72)
print(f"\n[Device] {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. THÉORIE – MODÈLES DE LANGAGE ET SÉQUENCES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("1. THÉORIE – MODÈLES DE LANGAGE ET ARCHITECTURES RÉCURRENTES")
print("─" * 72)

print("""
┌──────────────────────────────────────────────────────────────────────────┐
│  1.1 Objectif probabiliste d'un modèle de langage                        │
│  ─────────────────────────────────────────────────                       │
│  Un modèle de langage estime la probabilité d'une séquence de tokens :   │
│    P(x₁, x₂, …, xₜ)                                                     │
│                                                                          │
│  Par la règle de chaîne (factorisation auto-régressive) :                │
│    P(x₁,…,xₜ) = ∏ₜ P(xₜ | x₁,…,xₜ₋₁)                                 │
│                                                                          │
│  À chaque pas t, le modèle prédit la distribution sur le vocabulaire V   │
│  conditionnellement à tout le contexte précédent.                        │
│  Entraînement : minimiser la cross-entropie (= log-vraisemblance négative│
│    ℒ = -1/T Σₜ log P(xₜ | x₁,…,xₜ₋₁)                                  │
│                                                                          │
│  1.2 Perplexité                                                          │
│  ─────────────                                                           │
│  PPL = exp(ℒ) = exp(-1/T Σₜ log P(xₜ|contexte))                        │
│  Interprétation : nombre moyen de choix équiprobables à chaque token.   │
│  • PPL = 1   → prédiction parfaite                                      │
│  • PPL = |V| → modèle aléatoire (pire cas)                              │
│  • Comparaison : GPT-2 small → PPL ≈ 29 sur WikiText-103                │
│                                                                          │
│  1.3 Rétropropagation à travers le temps (BPTT)                         │
│  ─────────────────────────────────────────────                          │
│  Le gradient de la loss au pas T remonte à travers toutes les étapes :  │
│    ∂ℒ/∂h₁ = ∂ℒ/∂hₜ · ∏ₖ₌₁ᵀ (∂hₖ/∂hₖ₋₁)                              │
│  Si |∂h/∂h| < 1 à chaque pas → gradient disparaît (vanishing)          │
│  Si |∂h/∂h| > 1 → gradient explose (exploding)                         │
│  Remède : gradient clipping (∥g∥ > seuil → g ← g·seuil/∥g∥)           │
│                                                                          │
│  1.4 LSTM – Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)     │
│  ───────────────────────────────────────────────────────────────────    │
│  Cellule LSTM : état de cellule cₜ + état caché hₜ                      │
│    fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)   ← porte d'oubli (forget gate)         │
│    iₜ = σ(Wi·[hₜ₋₁,xₜ] + bi)   ← porte d'entrée (input gate)         │
│    g̃ₜ = tanh(Wg·[hₜ₋₁,xₜ]+bg)  ← candidat cellule                    │
│    cₜ = fₜ⊙cₜ₋₁ + iₜ⊙g̃ₜ        ← mise à jour cellule                 │
│    oₜ = σ(Wo·[hₜ₋₁,xₜ] + bo)   ← porte de sortie (output gate)       │
│    hₜ = oₜ⊙tanh(cₜ)             ← état caché sortant                  │
│                                                                          │
│  1.5 GRU – Gated Recurrent Unit (Cho et al., 2014)                      │
│  ──────────────────────────────────────────────────                     │
│  Simplification du LSTM : pas d'état de cellule séparé.                 │
│    zₜ = σ(Wz·[hₜ₋₁,xₜ])        ← porte de mise à jour (update gate)   │
│    rₜ = σ(Wr·[hₜ₋₁,xₜ])        ← porte de reset                      │
│    h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁,xₜ])  ← candidat caché                      │
│    hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ   ← état caché final                   │
│  GRU ≈ LSTM en performance mais avec 25% moins de paramètres.           │
└──────────────────────────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORPUS ET PRÉPARATION DES DONNÉES (MODÉLISATION DE LANGAGE)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 72)
print("2. PRÉPARATION DES DONNÉES – CORPUS INTÉGRÉ")
print("─" * 72)

# ── Corpus intégré (texte en anglais, domaine public) ────────────────────
CORPUS = """
the quick brown fox jumps over the lazy dog and the dog barked at the fox
machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed for every task
deep learning uses neural networks with many layers to learn representations
recurrent neural networks are designed to work with sequential data such as text
long short term memory networks solve the vanishing gradient problem in rnn
gated recurrent units are a simplified version of lstm with fewer parameters
the encoder reads the input sequence and produces a context vector
the decoder generates the output sequence token by token from the context
attention mechanisms allow the model to focus on relevant parts of the input
transformers use self attention instead of recurrence for sequence modeling
the training loss decreases as the model learns to predict the next token
perplexity measures how well a language model predicts a sample of text
gradient clipping prevents the gradients from exploding during backpropagation
the vanishing gradient problem makes it hard to learn long range dependencies
batch normalization helps stabilize training by normalizing layer activations
dropout randomly sets some activations to zero to prevent overfitting
the softmax function converts logits into a probability distribution over tokens
cross entropy loss measures the difference between predicted and true distributions
adam optimizer adapts the learning rate for each parameter individually
the embedding layer maps discrete tokens to dense continuous vectors
teacher forcing uses ground truth tokens as decoder inputs during training
beam search explores multiple hypotheses simultaneously during decoding
greedy decoding always picks the most probable next token at each step
the bleu score measures the overlap between generated and reference translations
sequence to sequence models encode a source sequence and decode a target sequence
natural language processing deals with understanding and generating human language
language models can be evaluated using perplexity on held out test data
the hidden state of a recurrent network summarizes all past information
bidirectional rnns process sequences in both forward and backward directions
the context vector bottleneck limits the capacity of vanilla seq2seq models
attention was introduced to overcome the bottleneck of a fixed context vector
word embeddings capture semantic relationships between words in a vector space
training on large corpora allows language models to learn rich representations
the number of parameters in a neural network affects its capacity and speed
regularization techniques help prevent the model from memorizing training data
the learning rate controls how much the model weights change at each update
momentum helps the optimizer escape local minima during gradient descent
weight initialization affects the stability and speed of neural network training
tokenization splits raw text into a sequence of discrete tokens or subwords
the vocabulary size determines the dimensionality of the output projection layer
padding tokens are used to make sequences in a mini batch the same length
masking prevents the model from attending to or predicting padding positions
mini batches allow efficient parallel computation during training on a gpu
the number of recurrent layers and hidden units affect model expressiveness
stacking multiple recurrent layers creates a deeper sequence model
residual connections help gradients flow through deep recurrent networks
layer normalization normalizes across the feature dimension for each token
the forget gate in lstm decides which information to discard from the cell state
the input gate decides which new information to add to the cell state
the output gate controls what part of the cell state is exposed as hidden state
"""

print(f"  Corpus : {len(CORPUS.split())} mots, {len(set(CORPUS.split()))} mots uniques")

# ── Tokenisation par mot ──────────────────────────────────────────────────
tokens = CORPUS.lower().split()

# Vocabulaire avec tokens spéciaux
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

counter  = Counter(tokens)
vocab    = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + \
           [w for w, _ in counter.most_common()]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(vocab)

PAD_IDX = word2idx[PAD_TOKEN]
UNK_IDX = word2idx[UNK_TOKEN]
BOS_IDX = word2idx[BOS_TOKEN]
EOS_IDX = word2idx[EOS_TOKEN]

print(f"  Taille vocabulaire : {VOCAB_SIZE}")
print(f"  Tokens spéciaux : PAD={PAD_IDX}, UNK={UNK_IDX}, BOS={BOS_IDX}, EOS={EOS_IDX}")

# ── Encodage du corpus ────────────────────────────────────────────────────
encoded = [word2idx.get(w, UNK_IDX) for w in tokens]

# ── Dataset : fenêtres glissantes (langue modeling) ──────────────────────
SEQ_LEN = 20  # longueur de chaque séquence d'entrée

class LMDataset(Dataset):
    """
    Dataset pour la modélisation de langage.
    Pour chaque position i, X = tokens[i:i+SEQ_LEN], y = tokens[i+1:i+SEQ_LEN+1]
    (prédiction du token suivant à chaque position → teacher forcing implicite).
    """
    def __init__(self, data, seq_len):
        self.data    = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, i):
        x = self.data[i   : i + self.seq_len]
        y = self.data[i+1 : i + self.seq_len + 1]
        return x, y

# Découpage train / val / test (70 / 15 / 15)
n = len(encoded)
n_tr = int(n * 0.70)
n_v  = int(n * 0.15)
tr_data = encoded[:n_tr]
v_data  = encoded[n_tr:n_tr + n_v]
te_data = encoded[n_tr + n_v:]

lm_train = LMDataset(tr_data, SEQ_LEN)
lm_val   = LMDataset(v_data,  SEQ_LEN)
lm_test  = LMDataset(te_data, SEQ_LEN)

BATCH = 32
lm_tr_loader = DataLoader(lm_train, BATCH, shuffle=True,  drop_last=True)
lm_va_loader = DataLoader(lm_val,   BATCH, shuffle=False, drop_last=True)
lm_te_loader = DataLoader(lm_test,  BATCH, shuffle=False, drop_last=True)

print(f"\n  Split LM : train={len(lm_train)}, val={len(lm_val)}, test={len(lm_test)} fenêtres")
print(f"  Longueur séquence (SEQ_LEN) : {SEQ_LEN}")
print(f"  Taille batch : {BATCH}")

print("""
  Concept de masquage (padding mask) :
  ─────────────────────────────────────
  Dans un mini-batch, les séquences de longueurs différentes sont alignées
  avec des tokens <pad>. Un masque booléen indique les positions de padding
  pour éviter que la loss soit calculée sur ces positions.
    mask = (y != PAD_IDX)  →  perte calculée uniquement sur les vraies positions.

  Teacher forcing :
  ─────────────────
  Pendant l'entraînement du décodeur, on fournit le vrai token précédent
  (et non la prédiction) comme entrée à chaque pas. Cela stabilise et
  accélère l'entraînement, mais crée un décalage avec l'inférence.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARCHITECTURES : RNN, LSTM, GRU
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 72)
print("3. ARCHITECTURES RÉCURRENTES")
print("─" * 72)

class LMModel(nn.Module):
    """
    Modèle de langage récurrent générique.
    Paramétré par cell_type ∈ {"RNN", "LSTM", "GRU"}.

    Architecture :
      Embedding(VOCAB_SIZE, embed_dim)
      → Dropout(embed_drop)
      → RNN|LSTM|GRU(embed_dim, hidden_dim, n_layers, dropout)
      → Linear(hidden_dim, VOCAB_SIZE)
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 n_layers=2, cell_type="LSTM", dropout=0.3):
        super().__init__()
        self.cell_type  = cell_type
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=PAD_IDX)
        self.embed_drop = nn.Dropout(dropout)

        rnn_drop = dropout if n_layers > 1 else 0.0
        if cell_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=rnn_drop)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        # Initialisation des poids
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len) – indices des tokens
        hidden : état caché initial (None = zéros)
        retourne logits (batch, seq_len, vocab_size) + nouvel état caché
        """
        emb = self.embed_drop(self.embedding(x))      # (B, T, E)
        out, hidden = self.rnn(emb, hidden)            # (B, T, H)
        logits = self.fc(self.dropout(out))            # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size):
        """Initialise l'état caché à zéro."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        if self.cell_type == "LSTM":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def detach_hidden(self, hidden):
        """Détache le graphe computationnel (TBPTT – Truncated BPTT)."""
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Affichage des architectures
for ct in ["RNN", "LSTM", "GRU"]:
    m = LMModel(VOCAB_SIZE, cell_type=ct)
    print(f"  [{ct:4s}] Paramètres : {m.count_params():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOUCLE D'ENTRAÎNEMENT + BPTT + GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("4. BPTT ET GRADIENT CLIPPING")
print("─" * 72)

print("""
  BPTT (Backpropagation Through Time) :
  ──────────────────────────────────────
  Pour un RNN déroulé sur T pas de temps, le gradient remonte à travers
  toutes les étapes. La perte totale est :
    ℒ = Σₜ ℒₜ(xₜ, ŷₜ)
  Le gradient ∂ℒ/∂Wrec implique des produits de matrices Jacobien :
    ∂hₜ/∂h₀ = ∏ₜ (∂hₜ/∂hₜ₋₁)
  → BPTT tronqué (Truncated BPTT) : on coupe le graphe toutes les k étapes
    avec hidden.detach() pour éviter des séquences de gradient trop longues.

  Gradient Clipping :
  ───────────────────
  Si ∥∇W∥ > threshold, on reéchelonne : ∇W ← ∇W · threshold / ∥∇W∥
  Implémentation PyTorch : torch.nn.utils.clip_grad_norm_(params, max_norm)
""")

CLIP = 1.0  # seuil de gradient clipping

def train_lm(model, tr_loader, va_loader, epochs=15, lr=1e-3,
             clip=CLIP, label="", verbose=True):
    """
    Entraîne un modèle de langage récurrent.
    Illustre BPTT tronqué (hidden.detach()) et gradient clipping.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    hist = {"train_loss": [], "val_loss": [],
            "train_ppl":  [], "val_ppl":  [],
            "grad_norms": []}          # pour illustrer l'effet du clipping
    best_val_loss = float("inf")
    best_weights  = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss, total_tokens = 0.0, 0
        hidden = None

        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            # BPTT tronqué : détacher l'état caché entre mini-batches
            if hidden is not None:
                hidden = model.detach_hidden(hidden)

            optimizer.zero_grad()
            logits, hidden = model(Xb, hidden)         # (B,T,V)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            loss.backward()

            # Gradient clipping
            gn = nn.utils.clip_grad_norm_(model.parameters(), clip)
            hist["grad_norms"].append(float(gn))

            optimizer.step()

            n_tok = (yb != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

        tr_loss = total_loss / max(total_tokens, 1)
        tr_ppl  = math.exp(min(tr_loss, 20))

        # ── Val ────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_tok = 0.0, 0
        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(Xb)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
                n_tok  = (yb != PAD_IDX).sum().item()
                v_loss += loss.item() * n_tok
                v_tok  += n_tok

        v_loss = v_loss / max(v_tok, 1)
        v_ppl  = math.exp(min(v_loss, 20))

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(v_loss)
        hist["train_ppl"].append(tr_ppl)
        hist["val_ppl"].append(v_ppl)

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights  = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{label}] Ép {epoch:2d}/{epochs} | "
                  f"Train loss={tr_loss:.4f} ppl={tr_ppl:.1f} | "
                  f"Val loss={v_loss:.4f} ppl={v_ppl:.1f}")

    model.load_state_dict(best_weights)
    elapsed = time.time() - t0
    print(f"  [{label}] ✓  Best val ppl={math.exp(min(best_val_loss,20)):.2f}  "
          f"(temps : {elapsed:.1f}s)")
    return model, hist, best_val_loss


def eval_ppl(model, loader):
    """Calcule la perplexité sur un DataLoader."""
    model.eval()
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits, _ = model(Xb)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            n = (yb != PAD_IDX).sum().item()
            total_loss += loss.item() * n
            total_tok  += n
    avg_loss = total_loss / max(total_tok, 1)
    return math.exp(min(avg_loss, 20))


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT ET COMPARAISON RNN / LSTM / GRU
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("5. ENTRAÎNEMENT ET COMPARAISON RNN / LSTM / GRU")
print("─" * 72)

EPOCHS_LM = 20
EMBED_DIM  = 64
HIDDEN_DIM = 128
N_LAYERS   = 2

results_lm = {}
models_lm  = {}

for cell_type in ["RNN", "LSTM", "GRU"]:
    print(f"\n▶ {cell_type}")
    m = LMModel(VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                n_layers=N_LAYERS, cell_type=cell_type, dropout=0.3)
    m, hist, best_loss = train_lm(
        m, lm_tr_loader, lm_va_loader,
        epochs=EPOCHS_LM, lr=1e-3, label=cell_type, verbose=True)

    test_ppl = eval_ppl(m, lm_te_loader)
    results_lm[cell_type] = {"hist": hist, "test_ppl": test_ppl,
                              "params": m.count_params()}
    models_lm[cell_type] = m

print("\n  ─── Tableau comparatif ─────────────────────────────────────────")
print(f"  {'Modèle':6s} | {'Test PPL':>10s} | {'Params':>10s} | "
      f"{'Stabilité':>12s}")
print("  " + "-" * 55)
for ct, r in results_lm.items():
    ppl_hist = r["hist"]["val_ppl"]
    if len(ppl_hist) > 1:
        stability = np.std(ppl_hist[-5:]) if len(ppl_hist) >= 5 else np.std(ppl_hist)
    else:
        stability = 0.0
    print(f"  {ct:6s} | {r['test_ppl']:>10.2f} | {r['params']:>10,} | "
          f"{stability:>12.4f}")

print("""
  Analyse comparative :
  ─────────────────────
  • RNN simple : gradients souvent instables sur longues séquences.
    Mémorisation limitée du contexte à longue distance.
  • LSTM : portes adaptatives → mémorisation longue distance efficace,
    gradient clipping moins critique. Converge plus stablement.
  • GRU : performances proches du LSTM, 25% moins de paramètres.
    Entraînement plus rapide. Préféré quand les ressources sont limitées.
  • Le gradient clipping (seuil=1.0) est essentiel pour les RNN simples
    où les gradient norms dépassent fréquemment 5-10.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ILLUSTRATION DE L'EFFET DU GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 72)
print("6. ILLUSTRATION EXPÉRIMENTALE – GRADIENT CLIPPING")
print("─" * 72)

print("\n  Entraînement sans gradient clipping (RNN) pour comparaison...")

def train_lm_noclip(model, tr_loader, epochs=5, lr=1e-3):
    """Variante sans gradient clipping – pour illustrer l'explosion."""
    model = copy.deepcopy(model).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    norms = []
    model.train()
    for epoch in range(epochs):
        hidden = None
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if hidden is not None:
                hidden = model.detach_hidden(hidden)
            opt.zero_grad()
            logits, hidden = model(Xb, hidden)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            loss.backward()
            # Calcul de la norme AVANT clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            norms.append(math.sqrt(total_norm))
            opt.step()   # ← pas de clipping
    return norms

rnn_noclip = LMModel(VOCAB_SIZE, cell_type="RNN", n_layers=N_LAYERS)
norms_noclip = train_lm_noclip(rnn_noclip, lm_tr_loader, epochs=3)
norms_clip   = results_lm["RNN"]["hist"]["grad_norms"]

print(f"  Sans clipping – norme moy={np.mean(norms_noclip):.2f}  "
      f"max={np.max(norms_noclip):.2f}")
print(f"  Avec clipping – norme moy={np.mean(norms_clip):.2f}  "
      f"max={np.max(norms_clip):.2f}  (seuil={CLIP})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SEQ2SEQ ENCODEUR–DÉCODEUR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("7. SYSTÈME SEQ2SEQ – ENCODEUR / DÉCODEUR")
print("─" * 72)

print("""
  Tâche : inversion de séquences de chiffres
  ───────────────────────────────────────────
  Entrée (source) : "3 1 4 1 5 9 2 6"
  Sortie (cible)  : "6 2 9 5 1 4 1 3"

  Cette tâche synthétique nécessite de mémoriser toute la séquence source
  avant de générer la sortie → test direct de la capacité de mémoire.

  Architecture Seq2Seq :
  ──────────────────────
                           context vector
  [x₁,…,xₙ] → Encoder → hₙ ─────────────→ Decoder → [ŷ₁,…,ŷₘ]

  • Encodeur (LSTM) : lit la séquence source, produit l'état caché final.
  • Décodeur (LSTM) : génère la séquence cible token par token,
                      initialisé avec l'état caché de l'encodeur.
  • Teacher forcing : pendant l'entraînement, on passe le vrai ŷₜ₋₁
                      au décodeur (et non sa propre prédiction).
""")

# ── Vocabulaire Seq2Seq (chiffres 0-9 + tokens spéciaux) ─────────────────
S2S_PAD = 0
S2S_BOS = 1
S2S_EOS = 2
DIGITS   = list(range(10))  # 0-9
S2S_VOCAB_SIZE = 13          # 0-9 + PAD(0) + BOS(1) + EOS(2)

# Encodage : chiffre d → d+3  (pour laisser 0,1,2 aux tokens spéciaux)
def digit_encode(seq):
    return [d + 3 for d in seq]

def digit_decode(seq):
    return [t - 3 for t in seq if t >= 3]

# ── Génération du dataset synthétique ────────────────────────────────────
def make_seq2seq_dataset(n, min_len=4, max_len=8, seed=0):
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n):
        length = rng.randint(min_len, max_len + 1)
        src    = rng.randint(0, 10, length).tolist()
        tgt    = src[::-1]                             # inversion
        pairs.append((src, tgt))
    return pairs

class Seq2SeqDataset(Dataset):
    """
    Dataset Seq2Seq pour l'inversion de séquences.
    X = [BOS] + encode(src) + [EOS]
    Y = [BOS] + encode(tgt) + [EOS]
    """
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        src_enc  = [S2S_BOS] + digit_encode(src) + [S2S_EOS]
        tgt_enc  = [S2S_BOS] + digit_encode(tgt) + [S2S_EOS]
        return (torch.tensor(src_enc, dtype=torch.long),
                torch.tensor(tgt_enc, dtype=torch.long))

def collate_seq2seq(batch):
    """Padding dynamique pour mini-batches de longueurs variables."""
    srcs, tgts = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True,
                                        padding_value=S2S_PAD)
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True,
                                        padding_value=S2S_PAD)
    return src_pad, tgt_pad

train_pairs = make_seq2seq_dataset(5000, seed=0)
val_pairs   = make_seq2seq_dataset(1000, seed=1)
test_pairs  = make_seq2seq_dataset(500,  seed=2)

s2s_tr = DataLoader(Seq2SeqDataset(train_pairs), batch_size=64,
                    shuffle=True,  collate_fn=collate_seq2seq)
s2s_va = DataLoader(Seq2SeqDataset(val_pairs),   batch_size=64,
                    shuffle=False, collate_fn=collate_seq2seq)
s2s_te = DataLoader(Seq2SeqDataset(test_pairs),  batch_size=64,
                    shuffle=False, collate_fn=collate_seq2seq)

print(f"  Dataset Seq2Seq : {len(train_pairs)} train / {len(val_pairs)} val / "
      f"{len(test_pairs)} test")


# ── Encodeur ──────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encodeur LSTM.
    Lit la séquence source et produit l'état caché (h, c) final
    qui sert de vecteur de contexte pour le décodeur.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=S2S_PAD)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))  # (B, T_src, E)
        _, (h, c) = self.lstm(emb)               # h,c : (n_layers, B, H)
        return h, c


# ── Décodeur ──────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Décodeur LSTM.
    Génère la séquence cible token par token.
    À chaque pas : prend le token précédent + (h, c) → prédit le prochain token.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=S2S_PAD)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc        = nn.Linear(hidden_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, token, h, c):
        """
        token : (B,) – token courant
        h, c  : état caché LSTM
        retourne logits (B, V) + nouvel état (h', c')
        """
        emb = self.dropout(self.embedding(token.unsqueeze(1)))  # (B,1,E)
        out, (h, c) = self.lstm(emb, (h, c))                    # (B,1,H)
        logits = self.fc(out.squeeze(1))                        # (B, V)
        return logits, h, c


# ── Seq2Seq complet ───────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    """
    Système Seq2Seq (encodeur LSTM + décodeur LSTM) pour l'inversion.
    Supporte le teacher forcing pendant l'entraînement.
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64,
                 n_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, n_layers, dropout)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src                   : (B, T_src)
        tgt                   : (B, T_tgt)
        teacher_forcing_ratio : probabilité d'utiliser le vrai token précédent
        retourne outputs (B, T_tgt-1, V)
        """
        B, T_tgt = tgt.shape
        outputs  = torch.zeros(B, T_tgt - 1, self.vocab_size).to(device)

        # Encodage
        h, c = self.encoder(src)

        # Premier token du décodeur = <BOS>
        token = tgt[:, 0]

        for t in range(T_tgt - 1):
            logits, h, c = self.decoder(token, h, c)  # (B, V)
            outputs[:, t] = logits

            # Teacher forcing
            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            token  = tgt[:, t + 1] if use_tf else logits.argmax(dim=-1)

        return outputs

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Entraînement Seq2Seq ──────────────────────────────────────────────────
def train_seq2seq(model, tr_loader, va_loader, epochs=20, lr=1e-3, clip=1.0):
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    crit  = nn.CrossEntropyLoss(ignore_index=S2S_PAD)
    hist  = {"train_loss": [], "val_loss": []}
    best_loss, best_wts = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tt = 0.0, 0
        for src, tgt in tr_loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            out  = model(src, tgt, teacher_forcing_ratio=0.5)  # (B,T-1,V)
            loss = crit(out.reshape(-1, S2S_VOCAB_SIZE), tgt[:,1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            n   = (tgt[:,1:] != S2S_PAD).sum().item()
            tl += loss.item() * n; tt += n

        model.eval()
        vl, vt = 0.0, 0
        with torch.no_grad():
            for src, tgt in va_loader:
                src, tgt = src.to(device), tgt.to(device)
                out  = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = crit(out.reshape(-1, S2S_VOCAB_SIZE), tgt[:,1:].reshape(-1))
                n   = (tgt[:,1:] != S2S_PAD).sum().item()
                vl += loss.item() * n; vt += n

        tl /= max(tt,1); vl /= max(vt,1)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        sch.step(vl)
        if vl < best_loss:
            best_loss = vl; best_wts = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Seq2Seq] Ép {epoch:2d}/{epochs} | "
                  f"Train loss={tl:.4f} | Val loss={vl:.4f}")

    model.load_state_dict(best_wts)
    return model, hist


print("\n▶ Entraînement du système Seq2Seq")
s2s_model = Seq2Seq(S2S_VOCAB_SIZE, embed_dim=32, hidden_dim=64, n_layers=1)
print(f"  Paramètres Seq2Seq : {s2s_model.count_params():,}")
s2s_model, hist_s2s = train_seq2seq(s2s_model, s2s_tr, s2s_va, epochs=25)


# ─────────────────────────────────────────────────────────────────────────────
# 8. STRATÉGIES DE DÉCODAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("8. STRATÉGIES DE DÉCODAGE")
print("─" * 72)

print("""
  Décodage glouton (Greedy decoding) :
  ─────────────────────────────────────
  À chaque pas t, on choisit argmax P(yₜ | y<ₜ, src).
  Simple et rapide mais peut manquer la séquence globalement optimale.

  Beam search :
  ─────────────
  On maintient les k meilleures hypothèses (k = beam width) à chaque pas.
  À chaque étape on étend chaque hypothèse par tous les tokens possibles,
  on garde les k meilleures par score de log-probabilité cumulée.
  Compromis : qualité ↑ avec k mais coût O(k · |V| · T).
""")

def greedy_decode(model, src_seq, max_len=12):
    """
    Décodage glouton d'une séquence source.
    src_seq : liste d'entiers (indices encodés)
    retourne : liste d'entiers décodés
    """
    model.eval()
    src = torch.tensor([[S2S_BOS] + digit_encode(src_seq) + [S2S_EOS]],
                       dtype=torch.long).to(device)
    with torch.no_grad():
        h, c  = model.encoder(src)
        token = torch.tensor([S2S_BOS], dtype=torch.long).to(device)
        output = []
        for _ in range(max_len):
            logits, h, c = model.decoder(token, h, c)
            pred  = logits.argmax(dim=-1)
            if pred.item() == S2S_EOS:
                break
            output.append(pred.item())
            token = pred
    return digit_decode(output)


def beam_search_decode(model, src_seq, beam_width=3, max_len=12):
    """
    Décodage par beam search.
    Maintient les beam_width meilleures hypothèses à chaque pas.
    Chaque hypothèse : (log_score, token_list, h, c)
    """
    model.eval()
    src = torch.tensor([[S2S_BOS] + digit_encode(src_seq) + [S2S_EOS]],
                       dtype=torch.long).to(device)
    with torch.no_grad():
        h0, c0 = model.encoder(src)
        # Initialisation : BOS comme premier token
        start_tok = torch.tensor([S2S_BOS], dtype=torch.long).to(device)
        logits, h0, c0 = model.decoder(start_tok, h0, c0)
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

        # beam : (log_score, tokens[], h, c)
        topk_lp, topk_ids = log_probs.topk(beam_width)
        beams = [(topk_lp[i].item(), [topk_ids[i].item()], h0, c0)
                 for i in range(beam_width)]

        completed = []
        for _ in range(max_len - 1):
            candidates = []
            for score, toks, h, c in beams:
                if toks[-1] == S2S_EOS:
                    completed.append((score, toks))
                    continue
                tok_t  = torch.tensor([toks[-1]], dtype=torch.long).to(device)
                logits, h_new, c_new = model.decoder(tok_t, h, c)
                lp = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_lp2, topk_ids2 = lp.topk(beam_width)
                for i in range(beam_width):
                    candidates.append(
                        (score + topk_lp2[i].item(),
                         toks + [topk_ids2[i].item()],
                         h_new, c_new))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # Prendre la meilleure hypothèse (complète ou partielle)
        all_hyps = completed + [(s, t) for s, t, _, _ in beams]
        all_hyps.sort(key=lambda x: x[0], reverse=True)
        best_toks = all_hyps[0][1] if all_hyps else []

    # Retirer BOS et EOS éventuels
    result = [t for t in best_toks if t not in (S2S_BOS, S2S_EOS)]
    return digit_decode(result)


# ── Test sur quelques exemples ────────────────────────────────────────────
print("\n  Exemples de décodage :")
print(f"  {'Source':20s} | {'Référence':20s} | {'Greedy':20s} | {'Beam(k=3)':20s}")
print("  " + "-" * 85)

test_examples = [[1,2,3,4], [5,9,2,6,3], [7,0,1,8,4,2], [3,1,4,1,5,9]]
for src in test_examples:
    ref    = src[::-1]
    greedy = greedy_decode(s2s_model, src)
    beam   = beam_search_decode(s2s_model, src, beam_width=3)
    print(f"  {str(src):20s} | {str(ref):20s} | {str(greedy):20s} | {str(beam):20s}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. ÉVALUATION – PERPLEXITÉ ET BLEU
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("9. ÉVALUATION – PERPLEXITÉ ET BLEU")
print("─" * 72)

# ── Perplexité LM sur test ────────────────────────────────────────────────
print("\n  [Perplexité sur jeu de test]")
for ct, r in results_lm.items():
    ppl = eval_ppl(models_lm[ct], lm_te_loader)
    print(f"  {ct:5s} → Test PPL = {ppl:.2f}")

# ── BLEU simplifié ────────────────────────────────────────────────────────
def compute_bleu(hypotheses, references, max_n=4):
    """
    Calcule le score BLEU (1 à max_n grammes) entre hypothèses et références.
    Implémentation simplifiée sans smoothing.
    BLEU = BP · exp(Σ wₙ · log pₙ)
      où pₙ = précision des n-grammes
         BP = brevity penalty = min(1, exp(1 - r/h))
    """
    def get_ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

    total_hyp_len = 0
    total_ref_len = 0
    precision     = []

    for n in range(1, max_n + 1):
        match, total = 0, 0
        for hyp, ref in zip(hypotheses, references):
            hyp_ng = Counter(get_ngrams(hyp, n))
            ref_ng = Counter(get_ngrams(ref, n))
            clipped = {ng: min(cnt, ref_ng[ng]) for ng, cnt in hyp_ng.items()}
            match += sum(clipped.values())
            total += max(len(hyp) - n + 1, 0)
        precision.append(match / max(total, 1))

    # Calcul sur le corpus complet pour BP
    for hyp, ref in zip(hypotheses, references):
        total_hyp_len += len(hyp)
        total_ref_len += len(ref)

    bp   = min(1.0, math.exp(1 - total_ref_len / max(total_hyp_len, 1)))
    prec = [p for p in precision if p > 0]
    if not prec:
        return 0.0
    log_sum = sum(math.log(p) for p in prec) / max_n
    return bp * math.exp(log_sum) * 100

# Évaluation BLEU sur le jeu de test Seq2Seq
hyps_greedy, hyps_beam, refs_all = [], [], []
for src_seq, tgt_seq in test_pairs[:100]:
    refs_all.append(tgt_seq)
    hyps_greedy.append(greedy_decode(s2s_model, src_seq))
    hyps_beam.append(beam_search_decode(s2s_model, src_seq, beam_width=3))

bleu_greedy = compute_bleu(hyps_greedy, refs_all)
bleu_beam   = compute_bleu(hyps_beam,   refs_all)

# Accuracy exacte (séquence entière correcte)
exact_greedy = sum(h==r for h,r in zip(hyps_greedy, refs_all)) / len(refs_all)
exact_beam   = sum(h==r for h,r in zip(hyps_beam,   refs_all)) / len(refs_all)

print(f"""
  [BLEU et Accuracy exacte – Seq2Seq inversion sur 100 exemples test]
  ┌────────────────┬──────────┬──────────────┐
  │ Stratégie      │ BLEU (%) │ Exact Match  │
  ├────────────────┼──────────┼──────────────┤
  │ Greedy         │ {bleu_greedy:>8.2f} │ {exact_greedy:>12.4f} │
  │ Beam search(3) │ {bleu_beam:>8.2f} │ {exact_beam:>12.4f} │
  └────────────────┴──────────┴──────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 72)
print("10. GÉNÉRATION DES FIGURES")
print("─" * 72)

# ── Figure 1 : Comparaison RNN/LSTM/GRU ──────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle("Comparaison RNN / LSTM / GRU – Modélisation de langage",
              fontsize=13, fontweight="bold")

colors = {"RNN": "#4C72B0", "LSTM": "#DD8452", "GRU": "#55A868"}

ax = axes[0]
for ct, r in results_lm.items():
    ax.plot(r["hist"]["train_ppl"], color=colors[ct],
            label=f"{ct} Train", linewidth=2)
    ax.plot(r["hist"]["val_ppl"],   color=colors[ct],
            label=f"{ct} Val",   linewidth=2, linestyle="--")
ax.set_title("Perplexité pendant l'entraînement")
ax.set_xlabel("Époque"); ax.set_ylabel("Perplexité")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_ylim(0, min(max(r["hist"]["val_ppl"][-1] for r in results_lm.values()) * 2.5, 300))

ax = axes[1]
for ct, r in results_lm.items():
    ax.plot(r["hist"]["train_loss"], color=colors[ct],
            label=f"{ct} Train", linewidth=2)
    ax.plot(r["hist"]["val_loss"],   color=colors[ct],
            label=f"{ct} Val",   linewidth=2, linestyle="--")
ax.set_title("Cross-Entropy Loss")
ax.set_xlabel("Époque"); ax.set_ylabel("Loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
labels_bar = list(results_lm.keys())
ppls        = [results_lm[ct]["test_ppl"] for ct in labels_bar]
params_bar  = [results_lm[ct]["params"] / 1000 for ct in labels_bar]
x = np.arange(len(labels_bar)); w = 0.35
bars1 = ax.bar(x - w/2, ppls,       w, label="Test PPL",  color=[colors[ct] for ct in labels_bar])
ax2   = ax.twinx()
bars2 = ax2.bar(x + w/2, params_bar, w, label="Params (k)",
                color=[colors[ct] for ct in labels_bar], alpha=0.5)
ax.set_title("PPL Test vs Paramètres")
ax.set_xticks(x); ax.set_xticklabels(labels_bar)
ax.set_ylabel("Test PPL"); ax2.set_ylabel("Paramètres (k)")
ax.set_ylim(0, max(ppls)*1.5)
for b, v in zip(bars1, ppls):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f"{v:.1f}", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("partie3_rnn_comparaison.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("  → partie3_rnn_comparaison.png")

# ── Figure 2 : Gradient clipping ─────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Effet du Gradient Clipping (BPTT – RNN)",
              fontsize=12, fontweight="bold")

ax = axes[0]
show = min(len(norms_noclip), 200)
ax.plot(norms_noclip[:show], color="tomato", label="Sans clipping", alpha=0.8)
ax.axhline(CLIP, color="black", linestyle="--", label=f"Seuil={CLIP}", linewidth=1.5)
ax.set_title("Normes de gradient – Sans clipping")
ax.set_xlabel("Itération"); ax.set_ylabel("‖∇W‖₂")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, min(max(norms_noclip[:show]), 20))

ax = axes[1]
show2 = min(len(norms_clip), 200)
ax.plot(norms_clip[:show2], color="steelblue", label="Avec clipping", alpha=0.8)
ax.axhline(CLIP, color="black", linestyle="--", label=f"Seuil={CLIP}", linewidth=1.5)
ax.set_title("Normes de gradient – Avec clipping")
ax.set_xlabel("Itération"); ax.set_ylabel("‖∇W‖₂")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, max(norms_clip[:show2]) * 1.2 if norms_clip else 2)

plt.tight_layout()
plt.savefig("partie3_gradient_clipping.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("  → partie3_gradient_clipping.png")

# ── Figure 3 : Seq2Seq loss + BLEU ───────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("Système Seq2Seq – Entraînement et évaluation",
              fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(hist_s2s["train_loss"], label="Train loss", color="seagreen", lw=2)
ax.plot(hist_s2s["val_loss"],   label="Val loss",   color="seagreen", lw=2, ls="--")
ax.set_title("Loss Seq2Seq (CrossEntropy)")
ax.set_xlabel("Époque"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
strategies = ["Greedy", "Beam(k=3)"]
bleu_vals  = [bleu_greedy, bleu_beam]
bars = ax.bar(strategies, bleu_vals, color=["steelblue", "seagreen"], width=0.5)
ax.set_title("BLEU Score – Seq2Seq inversion")
ax.set_ylabel("BLEU (%)"); ax.set_ylim(0, 100)
for b, v in zip(bars, bleu_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
exact_vals = [exact_greedy * 100, exact_beam * 100]
bars = ax.bar(strategies, exact_vals, color=["steelblue", "seagreen"], width=0.5)
ax.set_title("Exact Match – Séquence entière correcte")
ax.set_ylabel("Exact Match (%)"); ax.set_ylim(0, 100)
for b, v in zip(bars, exact_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("partie3_seq2seq.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("  → partie3_seq2seq.png")

# ── Figure 4 : Architecture Seq2Seq (diagramme SVG via matplotlib) ────────
fig4, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("Architecture Seq2Seq Encodeur–Décodeur", fontsize=13, fontweight="bold")

# Encodeur
for i, tok in enumerate(["x₁","x₂","x₃","…","xₙ"]):
    ax.add_patch(FancyBboxPatch((0.2+i*0.85, 0.8), 0.7, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#AED6F1", edgecolor="steelblue"))
    ax.text(0.55+i*0.85, 1.15, tok, ha="center", va="center", fontsize=10)

for i in range(5):
    ax.add_patch(FancyBboxPatch((0.2+i*0.85, 2.0), 0.7, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#5DADE2", edgecolor="steelblue", alpha=0.9))
    ax.text(0.55+i*0.85, 2.45, "LSTM", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.annotate("", xy=(0.55+i*0.85, 2.0), xytext=(0.55+i*0.85, 1.5),
                arrowprops=dict(arrowstyle="->", color="steelblue"))
    if i < 4:
        ax.annotate("", xy=(1.05+i*0.85, 2.45), xytext=(0.9+i*0.85, 2.45),
                    arrowprops=dict(arrowstyle="->", color="navy"))

ax.text(2.4, 3.3, "Encodeur", ha="center", fontsize=11,
        color="steelblue", fontweight="bold")
ax.text(4.7, 3.3, "Contexte\n(hₙ, cₙ)", ha="center", fontsize=10,
        color="black",
        bbox=dict(boxstyle="round", facecolor="#F9E79F", edgecolor="orange"))

# Flèche context
ax.annotate("", xy=(6.5, 2.45), xytext=(5.3, 2.45),
            arrowprops=dict(arrowstyle="->", color="orange", lw=2))

# Décodeur
for i, tok in enumerate(["<BOS>","ŷ₁","ŷ₂","…","ŷₘ"]):
    ax.add_patch(FancyBboxPatch((6.3+i*0.9, 2.0), 0.75, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#A9DFBF", edgecolor="seagreen", alpha=0.9))
    ax.text(6.68+i*0.9, 2.45, "LSTM", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    # token input
    ax.text(6.68+i*0.9, 1.3, tok, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#D5F5E3", edgecolor="seagreen"))
    ax.annotate("", xy=(6.68+i*0.9, 2.0), xytext=(6.68+i*0.9, 1.6),
                arrowprops=dict(arrowstyle="->", color="seagreen"))
    if i < 4:
        ax.annotate("", xy=(7.2+i*0.9, 2.45), xytext=(7.05+i*0.9, 2.45),
                    arrowprops=dict(arrowstyle="->", color="darkgreen"))
    # output
    if i > 0:
        ax.add_patch(FancyBboxPatch((6.3+i*0.9, 3.1), 0.75, 0.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor="#FDFEFE", edgecolor="gray"))
        ax.text(6.68+i*0.9, 3.35, f"P(y{i})", ha="center", fontsize=8)
        ax.annotate("", xy=(6.68+i*0.9, 3.1), xytext=(6.68+i*0.9, 2.9),
                    arrowprops=dict(arrowstyle="->", color="gray"))

ax.text(9.3, 3.3, "Décodeur", ha="center", fontsize=11,
        color="seagreen", fontweight="bold")

plt.tight_layout()
plt.savefig("partie3_architecture.png", dpi=130, bbox_inches="tight")
plt.close()
print("  → partie3_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
best_ct = min(results_lm, key=lambda ct: results_lm[ct]["test_ppl"])
torch.save(models_lm[best_ct].state_dict(),
           f"best_lm_{best_ct.lower()}.pth")
torch.save(s2s_model.state_dict(),
           "best_seq2seq.pth")
print(f"\n  Meilleur LM ({best_ct}) sauvegardé → best_lm_{best_ct.lower()}.pth")
print("  Seq2Seq sauvegardé → best_seq2seq.pth")


# ─────────────────────────────────────────────────────────────────────────────
# 12. QUESTION DE SYNTHÈSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("11. QUESTION DE SYNTHÈSE")
print("─" * 72)

print(f"""
  « Dans quelle mesure les architectures récurrentes permettent-elles de
    modéliser efficacement une séquence réelle, et comment justifier le
    passage RNN → LSTM/GRU → encodeur–décodeur ? »

  ════════════════════════════════════════════════════════════════════════

  1. Modélisation probabiliste des séquences
  ───────────────────────────────────────────
  Un modèle de langage estime P(x₁,…,xₜ) = ∏ P(xₜ|x<ₜ) via la règle de
  chaîne. L'état caché hₜ d'un RNN résume le contexte x<ₜ de façon
  compressée. La perplexité PPL = exp(ℒ) mesure la qualité de cette
  estimation : PPL=1 (prédiction parfaite), PPL=|V| (aléatoire).

  Résultats sur notre corpus :
    RNN  → PPL = {results_lm["RNN"]["test_ppl"]:.2f}
    LSTM → PPL = {results_lm["LSTM"]["test_ppl"]:.2f}
    GRU  → PPL = {results_lm["GRU"]["test_ppl"]:.2f}

  2. Passage du RNN simple au LSTM/GRU
  ──────────────────────────────────────
  Le RNN simple souffre du problème du gradient qui disparaît/explose
  (BPTT) : le produit des Jacobiens ∏∂hₜ/∂hₜ₋₁ tend vers 0 ou ∞.
  En pratique, le RNN ne mémorise que ~10-20 pas d'histoire.

  Le gradient clipping (seuil=1.0) limite l'explosion mais ne résout pas
  la disparition. Nos courbes de gradient_norms montrent :
    Sans clipping : norme moy={np.mean(norms_noclip):.2f}, max={np.max(norms_noclip):.2f}
    Avec clipping : norme bornée à {CLIP}

  LSTM : l'état de cellule cₜ crée un "autoroute de gradient" où le
  gradient traverse le temps sans atténuation (fₜ≈1, iₜ≈0 → cₜ=cₜ₋₁).
  Les 3 portes adaptatives apprennent quoi oublier, stocker et révéler.

  GRU : simplification en 2 portes (reset + update), performances
  comparables au LSTM avec 25% de paramètres en moins. Préféré quand les
  ressources de calcul ou les données sont limitées.

  3. Justification du Seq2Seq encodeur–décodeur
  ───────────────────────────────────────────────
  Quand entrée et sortie n'ont pas la même longueur (traduction, résumé),
  un seul RNN ne suffit pas. Le paradigme Seq2Seq :
  • Encodeur : compresse la source en vecteur de contexte (hₙ, cₙ).
  • Décodeur : génère la cible conditionnellement à ce contexte.
  • Teacher forcing : réduit la propagation des erreurs à l'entraînement.

  Sur notre tâche d'inversion :
    BLEU greedy   = {bleu_greedy:.1f}%    Exact match = {exact_greedy*100:.1f}%
    BLEU beam(k=3)= {bleu_beam:.1f}%    Exact match = {exact_beam*100:.1f}%

  Le beam search améliore la qualité en explorant k hypothèses parallèles
  au lieu de s'engager greedy à chaque pas.

  4. Qualité du décodage et limites
  ────────────────────────────────────
  • Le vecteur de contexte fixe est un goulot d'étranglement : toute
    l'information source est compressée en un seul vecteur (h, c).
    Solution naturelle : le mécanisme d'attention (Bahdanau, 2015) qui
    crée un contexte dynamique α·Henc à chaque pas du décodeur.
  • Les Transformers (Vaswani, 2017) remplacent la récurrence par
    l'auto-attention, éliminant la dépendance séquentielle et permettant
    la parallélisation totale de l'entraînement.

  5. Conclusion
  ─────────────
  La progression RNN → LSTM/GRU → Seq2Seq est motivée par des problèmes
  identifiés expérimentalement (gradients, mémoire, longueurs variables).
  Chaque architecture apporte une solution ciblée : LSTM résout les
  gradients lointains, le Seq2Seq gère les longueurs asymétriques, et
  l'attention (prochaine étape naturelle) lève le goulot du contexte fixe.
""")

print("=" * 72)
print("FIN – PARTIE III COMPLÈTE")
print("=" * 72)
