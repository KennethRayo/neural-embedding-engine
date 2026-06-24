import pickle
import numpy as np
import sys

def load_data(path="embeddings.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró '{path}'. Ejecuta 'train.py' primero.")
        sys.exit(1)

def load_model(data):
    """Reconstruye el modelo desde el state_dict guardado (para fine-tuning o inferencia)."""
    try:
        from model.model import SkipGramModel
        import torch
        config = data["model_config"]
        model = SkipGramModel(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
        )
        model.load_state_dict(data["model_state_dict"])
        model.eval()
        return model
    except KeyError:
        return None

def main():
    # 1. Cargar embeddings y diccionarios
    data = load_data()
    embeddings = data["embeddings"]
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]

    # Informar si el checkpoint incluye el modelo completo
    has_model = "model_state_dict" in data
    if has_model:
        config = data.get("model_config", {})
        print(f"Checkpoint completo detectado — vocab: {config.get('vocab_size')}, dim: {config.get('embedding_dim')}")

    # 2. Normalizar todos los vectores para calcular similitud coseno
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normalized_embeddings = embeddings / norms

    def get_nearest_neighbors(word, top_k=5):
        word_clean = word.lower().strip()
        if word_clean not in word2idx:
            print(f"La palabra '{word}' no se encuentra en el vocabulario.")
            return

        idx = word2idx[word_clean]
        word_vector = normalized_embeddings[idx]

        similarities = np.dot(normalized_embeddings, word_vector)
        sorted_indices = np.argsort(similarities)[::-1]

        print(f"\nPalabras más similares a '{word_clean}':")
        count = 0
        for near_idx in sorted_indices:
            near_word = idx2word[near_idx]
            if near_word == word_clean or near_word == '<UNK>':
                continue
            print(f"  - {near_word:<15} | Similitud: {similarities[near_idx]:.4f}")
            count += 1
            if count >= top_k:
                break

    def get_analogy(w1, w2, w3, top_k=3):
        """
        Resuelve analogías: w1 es a w2 como w3 es a X
        Fórmula: vector(X) = vector(w1) - vector(w2) + vector(w3)
        """
        w1, w2, w3 = w1.lower().strip(), w2.lower().strip(), w3.lower().strip()
        for w in [w1, w2, w3]:
            if w not in word2idx:
                print(f"La palabra '{w}' no se encuentra en el vocabulario.")
                return

        idx1, idx2, idx3 = word2idx[w1], word2idx[w2], word2idx[w3]

        target_vec = normalized_embeddings[idx1] - normalized_embeddings[idx2] + normalized_embeddings[idx3]

        norm = np.linalg.norm(target_vec)
        if norm > 0:
            target_vec = target_vec / norm

        similarities = np.dot(normalized_embeddings, target_vec)
        sorted_indices = np.argsort(similarities)[::-1]

        print(f"\nAnalogía: '{w1}' es a '{w2}' como '{w3}' es a ...")
        count = 0
        for idx in sorted_indices:
            near_word = idx2word[idx]
            if near_word in [w1, w2, w3, '<UNK>']:
                continue
            print(f"  - {near_word:<15} | Similitud: {similarities[idx]:.4f}")
            count += 1
            if count >= top_k:
                break

    # Menú de interacción
    while True:
        print("\n=======================================================")
        print("   EVALUADOR INTERACTIVO DE EMBEDDINGS (Word2Vec)      ")
        print("=======================================================")
        print("1. Buscar palabras similares (Similitud Coseno)")
        print("2. Resolver analogía (ej: rey es a hombre como reina es a...)")
        print("3. Ejecutar pruebas predefinidas")
        print("4. Mostrar palabras en el vocabulario")
        print("5. Salir")
        print("=======================================================")

        try:
            opcion = input("Elige una opción (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            break

        if opcion == "1":
            word = input("Palabra a buscar: ").strip()
            get_nearest_neighbors(word)
        elif opcion == "2":
            print("\nFórmula: A es a B como C es a X (Ej: perro es a ladra como gato es a...)")
            w1 = input("Palabra A: ").strip()
            w2 = input("Palabra B: ").strip()
            w3 = input("Palabra C: ").strip()
            if w1 and w2 and w3:
                get_analogy(w1, w2, w3)
        elif opcion == "3":
            test_words = ["software", "código", "programación", "gato", "perro", "animales"]
            for word in test_words:
                get_nearest_neighbors(word, top_k=4)
        elif opcion == "4":
            words = sorted([w for w in word2idx.keys() if w != '<UNK>'])
            print(f"\nVocabulario ({len(words)} palabras):")
            col_width = 18
            cols = 4
            for i in range(0, len(words), cols):
                chunk = words[i:i+cols]
                print("".join(f"{w:<{col_width}}" for w in chunk))
        elif opcion == "5":
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
