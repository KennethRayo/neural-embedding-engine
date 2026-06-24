import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    """
    Modelo Skip-Gram con Negative Sampling (SGNS) implementado desde cero.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Matriz de embeddings para las palabras objetivo (Input)
        self.target_embeddings = nn.Embedding(num_embeddings=vocab_size, 
                                              embedding_dim=embedding_dim)
        
        # Matriz de embeddings para las palabras de contexto (Output/Noise)
        self.context_embeddings = nn.Embedding(num_embeddings=vocab_size, 
                                               embedding_dim=embedding_dim)
        
        # Inicializar los pesos de manera uniforme
        initrange = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.fill_(0.0)
        
    def forward(self, target, context, negative):
        # target: (batch_size)
        # context: (batch_size)
        # negative: (batch_size, num_negatives)
        
        # 1. Obtener representaciones vectoriales
        emb_target = self.target_embeddings(target)        # (batch_size, embedding_dim)
        emb_context = self.context_embeddings(context)      # (batch_size, embedding_dim)
        emb_negative = self.context_embeddings(negative)    # (batch_size, num_negatives, embedding_dim)
        
        # 2. Calcular puntaje positivo: producto escalar entre target y contexto
        pos_score = torch.sum(emb_target * emb_context, dim=1)  # (batch_size)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)
        
        # 3. Calcular puntaje negativo: producto escalar entre target y las palabras negativas
        # Multiplicación de matrices por lotes (BMM)
        # (batch_size, 1, embedding_dim) x (batch_size, embedding_dim, num_negatives) -> (batch_size, 1, num_negatives)
        neg_score = torch.bmm(emb_target.unsqueeze(1), emb_negative.transpose(1, 2)).squeeze(1) # (batch_size, num_negatives)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-9), dim=1)  # (batch_size)
        
        # Maximizar similitud con positivos y minimizar con negativos es equivalente a minimizar la suma negativa
        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        """
        Retorna la matriz de embeddings aprendida para las palabras de entrada.
        """
        return self.target_embeddings.weight.data