import torch
from torch.utils.data import Dataset
import collections
import re
import numpy as np

class Vocabulary:
    """
    Clase para construir y manejar el vocabulario de un corpus de texto.
    """
    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_freqs = collections.Counter()
        
    def build_vocab(self, sentences):
        # 1. Contar frecuencias de todas las palabras
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            self.word_freqs.update(tokens)
            
        # 2. Agregar token especial para palabras desconocidas
        self.word2idx['<UNK>'] = 0
        self.idx2word[0] = '<UNK>'
        
        # 3. Construir los mapeos filtrando por frecuencia mínima
        idx = 1
        for word, count in self.word_freqs.items():
            if count >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def tokenize(self, text):
        # Limpieza básica: minúsculas y eliminación de caracteres no alfanuméricos
        text = text.lower()
        return re.findall(r'\b\w+\b', text)
        
    def __len__(self):
        return len(self.word2idx)
        
    def get_idx(self, word):
        return self.word2idx.get(word, self.word2idx['<UNK>'])
        
    def get_word(self, idx):
        return self.idx2word.get(idx, '<UNK>')


class Word2VecDataset(Dataset):
    """
    Dataset optimizado para Skip-Gram con Negative Sampling (SGNS) a partir de un corpus de texto.
    """
    def __init__(self, sentences, min_count=1, window_size=2, num_negatives=5, subsample_threshold=1e-4):
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.subsample_threshold = subsample_threshold
        
        # Construir el vocabulario
        self.vocab = Vocabulary(min_count=min_count)
        self.vocab.build_vocab(sentences)
        
        # Indexar las frases
        self.indexed_sentences = []
        for sentence in sentences:
            tokens = self.vocab.tokenize(sentence)
            indices = [self.vocab.get_idx(t) for t in tokens]
            if len(indices) > 1:  # Ignorar frases vacías o de una sola palabra
                self.indexed_sentences.append(indices)
                
        # Aplicar submuestreo (subsampling) de palabras frecuentes para reducir ruido
        self.subsampled_sentences = self._subsample_sentences()
                
        # Generar parejas (palabra_objetivo, palabra_contexto)
        self.pairs = []
        for sentence in self.subsampled_sentences:
            for i, target_idx in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        self.pairs.append((target_idx, sentence[j]))
                        
        # Construir la distribución para Negative Sampling: P(w) ~ freq(w)^0.75
        self.neg_sampler = self._build_negative_sampler()
        
        # Precalcular una tabla grande de muestreo negativo para optimizar __getitem__
        self.neg_table = self._build_negative_table(size=1_000_000)
        
    def _subsample_sentences(self):
        # 1. Contar total de palabras en oraciones indexadas
        total_words = sum(len(s) for s in self.indexed_sentences)
        if total_words == 0:
            return self.indexed_sentences
            
        # 2. Calcular frecuencia de cada palabra indexada
        freqs = collections.Counter()
        for sentence in self.indexed_sentences:
            freqs.update(sentence)
            
        subsampled = []
        for sentence in self.indexed_sentences:
            keep_sentence = []
            for word_idx in sentence:
                word = self.vocab.get_word(word_idx)
                if word == '<UNK>':
                    keep_sentence.append(word_idx)
                    continue
                # Frecuencia relativa
                f = freqs[word_idx] / total_words
                # Fórmula de Mikolov para determinar probabilidad de mantener la palabra
                keep_prob = (np.sqrt(f / self.subsample_threshold) + 1) * (self.subsample_threshold / f)
                keep_prob = min(1.0, keep_prob)
                
                if np.random.random() < keep_prob:
                    keep_sentence.append(word_idx)
            if len(keep_sentence) > 1:
                subsampled.append(keep_sentence)
        return subsampled

    def _build_negative_sampler(self):
        weights = np.zeros(len(self.vocab))
        for word, idx in self.vocab.word2idx.items():
            if word == '<UNK>':
                count = 1
            else:
                count = self.vocab.word_freqs[word]
            weights[idx] = count ** 0.75
            
        weights = weights / np.sum(weights)
        return weights

    def _build_negative_table(self, size):
        # Muestrear una tabla grande una sola vez en lugar de hacerlo por muestra
        return np.random.choice(len(self.vocab), size=size, p=self.neg_sampler)
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        target_idx, context_idx = self.pairs[idx]
        
        # Muestreo negativo ultra-rápido usando la tabla precalculada
        neg_samples = []
        ptr = (idx * self.num_negatives * 2) % (len(self.neg_table) - self.num_negatives * 2)
        candidates = self.neg_table[ptr : ptr + self.num_negatives * 2]
        
        for s_idx in candidates:
            if s_idx != target_idx and s_idx != context_idx and s_idx not in neg_samples:
                neg_samples.append(s_idx)
            if len(neg_samples) == self.num_negatives:
                break
                
        # Método de respaldo en caso de excesivas colisiones en vocabularios minúsculos
        while len(neg_samples) < self.num_negatives:
            s_idx = int(np.random.choice(len(self.vocab), p=self.neg_sampler))
            if s_idx != target_idx and s_idx != context_idx and s_idx not in neg_samples:
                neg_samples.append(s_idx)
                    
        return (
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long)
        )