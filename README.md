# Neural Embedding Engine

Implementación desde cero de **Word2Vec** (Skip-Gram con Negative Sampling) usando PyTorch puro. Sin gensim, sin fastText — cada componente construido a mano para entender los embeddings de palabras en profundidad.

## Arquitectura

```
corpus.txt  →  Word2VecDataset  →  SkipGramModel  →  embeddings.pkl
                  (vocab +             (2 matrices          (numpy +
                subsampling +          embedding)          state_dict)
                neg. sampling)
```

El modelo aprende a predecir palabras de contexto a partir de una palabra objetivo (Skip-Gram). La función de pérdida es *negative log-sigmoid* (BCE), optimizada con Negative Sampling para evitar calcular el softmax sobre todo el vocabulario.

## Estructura del proyecto

```
neural-embedding-engine/
├── data/
│   └── corpus.txt          # Corpus de entrenamiento (español)
├── dataset/
│   └── dataset.py          # Vocabulary + Word2VecDataset
├── model/
│   └── model.py            # SkipGramModel (PyTorch)
├── train.py                # Bucle de entrenamiento
├── eval.py                 # Evaluador interactivo CLI
├── embeddings.pkl          # Artefacto generado (ignorado por git)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### Entrenamiento

```bash
python train.py
```

Genera `embeddings.pkl` con los vectores aprendidos, el vocabulario y el `state_dict` del modelo.

```
Usando dispositivo: cpu
Cargando corpus...
Vocabulario cargado: 187 palabras únicas.
Época 001/100 | Pérdida Promedio: 3.4521 | LR: 0.003000
Época 010/100 | Pérdida Promedio: 2.1834 | LR: 0.002700
...
¡Entrenamiento completo!
```

### Evaluación interactiva

```bash
python eval.py
```

```
=======================================================
   EVALUADOR INTERACTIVO DE EMBEDDINGS (Word2Vec)
=======================================================
1. Buscar palabras similares (Similitud Coseno)
2. Resolver analogía (ej: rey es a hombre como reina es a...)
3. Ejecutar pruebas predefinidas
4. Mostrar palabras en el vocabulario
5. Salir
```

**Similitud coseno** — encuentra las palabras más cercanas a una consulta:

```
> software
  - código          | Similitud: 0.8921
  - programación    | Similitud: 0.8714
  - desarrollador   | Similitud: 0.8502
```

**Analogías** — resuelve relaciones semánticas mediante aritmética vectorial (`A - B + C`):

```
> perro : ladra :: gato : ?
  - maulla          | Similitud: 0.7643
```

## Hiperparámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| `EMB_DIM` | 50 | Dimensión del vector de embedding |
| `WINDOW_SIZE` | 3 | Tamaño de la ventana de contexto |
| `NUM_NEGATIVES` | 5 | Muestras negativas por par positivo |
| `LR` | 0.003 | Tasa de aprendizaje inicial |
| `EPOCHS` | 100 | Épocas de entrenamiento |
| `BATCH_SIZE` | 32 | Tamaño del lote |

El learning rate decae linealmente (`1 - epoch/EPOCHS`) usando `LambdaLR`.

## Detalles de implementación

**Negative Sampling** — la distribución de muestreo sigue la fórmula de Mikolov: `P(w) ∝ freq(w)^0.75`, que suaviza la frecuencia y da más oportunidad a palabras raras. Se precalcula una tabla de 1M entradas para muestreo O(1) en `__getitem__`.

**Subsampling** — las palabras muy frecuentes se descartan con probabilidad proporcional a su frecuencia relativa (fórmula original de Word2Vec), reduciendo el ruido en corpus con stopwords.

**Inicialización de pesos** — los embeddings de entrada se inicializan con distribución uniforme `±(0.5 / dim)` y los de salida a cero, práctica estándar del paper original.

**GPU** — el entrenamiento detecta automáticamente CUDA y mueve el modelo y los tensores al dispositivo disponible.

## Corpus

El corpus incluye ~100 frases en español distribuidas en tres dominios semánticos:

- **Tecnología / programación** — software, Python, redes neuronales, embeddings
- **Animales / mascotas** — perro, gato, veterinario, mascota
- **Ciencia / naturaleza** — física, biología, universo, ecosistemas

Esta separación permite verificar que el modelo aprende clusters semánticos diferenciados.

## Licencia

MIT — ver [LICENSE](LICENSE).
