import torch
import torch.nn as nn


class PTUMTransformerModel(nn.Module):
    def __init__(self, num_animes, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(PTUMTransformerModel, self).__init__()

        # Anime embedding layer
        self.anime_embedding = nn.Embedding(num_animes + 1, embed_dim)

        # User behavior encoder (transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = hidden_dim,
            dropout = dropout,
            batch_first = True
        )
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def calculate_mean_embedding(self, user_embeddings, padding_mask):

        # Masking out padded entries and computing the mean over non-padded elements
        mask = padding_mask.unsqueeze(-1).to(torch.float32)  # [batch_size, seq_len, 1]
        user_embeddings = user_embeddings * mask  # Zero out padded embeddings

        # Compute the sum of embeddings over non-padded positions
        sum_embeddings = user_embeddings.sum(dim=1)  # [batch_size, embed_dim]
        num_non_padded = mask.sum(dim=1)  # [batch_size, 1]

        # Compute the mean over non-padded elements
        user_embeddings = sum_embeddings / num_non_padded  # [batch_size, embed_dim]

        return user_embeddings

    def forward(self, behavior_sequences, candidate_sets, padding_mask):
        # Encode user behavior sequence into a single user embedding
        behavior_embeddings = self.anime_embedding(behavior_sequences)  # [batch_size, seq_len, embed_dim]
        user_embeddings = self.user_encoder(behavior_embeddings, src_key_padding_mask=~padding_mask)
        user_embeddings = self.calculate_mean_embedding(user_embeddings, padding_mask)

        # Encode candidate behaviors

        if len(candidate_sets.shape) == 2:
          batch_size, P_plus_1 = candidate_sets.shape
          candidate_sets = candidate_sets.view(batch_size, 1, P_plus_1)
        batch_size, K, P_plus_1 = candidate_sets.shape
        candidate_embeddings = self.anime_embedding(candidate_sets.view(batch_size * K, P_plus_1))  # [batch_size*K, P+1, embed_dim]

        # Compute scores for each candidate
        scores = torch.bmm(candidate_embeddings, user_embeddings.unsqueeze(-1).repeat(K, 1, 1)).squeeze(-1)  # [batch_size*K, P+1]
        return scores.view(batch_size, K, P_plus_1)  # Reshape to [batch_size, K, P+1]