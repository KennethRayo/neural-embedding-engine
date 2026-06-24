import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import SkipGramModel
from dataset.dataset import Word2VecDataset
import pickle

# 1. Configuración de hiperparámetros
EMB_DIM = 50          # Dimensión del vector de embedding
WINDOW_SIZE = 3      # Tamaño de la ventana de contexto
NUM_NEGATIVES = 5    # Número de palabras negativas por ejemplo positivo
LR = 0.003            # Tasa de aprendizaje
EPOCHS = 100         # Épocas de entrenamiento
BATCH_SIZE = 32      # Tamaño del lote

# 2. Selección de dispositivo (GPU si está disponible, si no CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 3. Cargar y preparar el corpus de texto
print("Cargando corpus...")
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Inicializar dataset y cargador de datos
dataset = Word2VecDataset(sentences, min_count=1, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES, subsample_threshold=1e-2)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vocab_size = len(dataset.vocab)
print(f"Vocabulario cargado: {vocab_size} palabras únicas.")
print(f"Número de pares target-contexto generados: {len(dataset)}")

# 4. Inicialización del modelo, optimizador y planificador de aprendizaje
model = SkipGramModel(vocab_size=vocab_size, embedding_dim=EMB_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Planificador de decaimiento lineal del learning rate
lr_lambda = lambda epoch: 1.0 - (epoch / EPOCHS)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# 5. Bucle de Entrenamiento
print("Iniciando entrenamiento del modelo Word2Vec desde cero...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for targets, contexts, negatives in dataloader:
        targets = targets.to(device)
        contexts = contexts.to(device)
        negatives = negatives.to(device)

        optimizer.zero_grad()
        loss = model(targets, contexts, negatives)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Época {epoch+1:03d}/{EPOCHS} | Pérdida Promedio: {total_loss/len(dataloader):.4f} | LR: {current_lr:.6f}")

# 6. Extraer y guardar embeddings, vocabulario y modelo completo
embeddings = model.get_embeddings()

data_to_save = {
    "embeddings": embeddings.cpu().numpy(),
    "word2idx": dataset.vocab.word2idx,
    "idx2word": dataset.vocab.idx2word,
    "model_state_dict": model.state_dict(),
    "model_config": {
        "vocab_size": vocab_size,
        "embedding_dim": EMB_DIM,
    },
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print("\n¡Entrenamiento completo!")
print("Embeddings, vocabulario y modelo guardados en 'embeddings.pkl'")
