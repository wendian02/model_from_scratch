import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, L):
        super().__init__()
        self.embed_dim = embed_dim
        self.L = L
    
    def forward(self):
        embed_dim, L = self.embed_dim, self.L
        even_i = torch.arange(0, embed_dim, 2).reshape(1,-1)
        pos = torch.arange(L).reshape(-1,1)
        PE_even = torch.sin(pos / (10000 ** (even_i / embed_dim))) # L x (E/2)
        PE_odd = torch.cos(pos / (10000 ** (even_i / embed_dim))) # L x (E/2)
        PE_stacked = torch.stack([PE_even, PE_odd], dim=2) # L x (E/2) x 2
        PE = PE_stacked.reshape(L, embed_dim) # L x E
        # PE = torch.flatten(PE_stacked, start_dim=1, end_dim=2)
        return PE


def scaled_dot_product_attention(q, k, v, mask=None): # q, k, v: B x n_heads x L x (E // n_heads)
    d_k = q.shape[-1]
    scale_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # B x n_heads x L x L
    if mask is not None:
        scale_score += mask.unsqueeze(1) # -inf for masked positions mask before unsqueeze: B x L x L

    attn_weights = F.softmax(scale_score, dim=-1) # B x n_heads x L x L
    context_v = torch.matmul(attn_weights, v) # B x n_heads x L x (E // n_heads)
    return context_v, attn_weights

class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = nn.Parameter(torch.ones(embed_dim)) # embed_dim,
        self.beta = nn.Parameter(torch.zeros(embed_dim)) 
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # B x L x 1
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True) # B x L x 1
        std = (var + self.eps).sqrt() 
        x_norm = (x - mean) / std
        return self.gamma * x_norm + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, E = x.shape # B x L x E
        qkv = self.qkv(x) # B x L x 3E
        qkv = qkv.reshape(B, L, self.n_heads, 3 * self.embed_dim // self.n_heads) # B x L x n_heads x (3E // n_heads)
        qkv = qkv.permute(0, 2, 1, 3) # B x n_heads x L  x (3E // n_heads)
        q, k, v = qkv.chunk(3, dim=-1) # B x n_heads x L x (E // n_heads)
        # q, k, v = qkv.split(self.embed_dim // self.n_heads, dim=-1) # B x n_heads x L x (E // n_heads)
        context_v, _ = scaled_dot_product_attention(q, k, v, mask) # B x n_heads x L x (E // n_heads)
        context_v_concat = context_v.permute(0, 2, 1, 3) # B x L x n_heads x (E // n_heads)
        context_v_concat = context_v_concat.reshape(B, L, E) # B x L x E
        context_v_concat = self.linear(context_v_concat) # B x L x E
        return context_v_concat


class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y, mask=None):
        B, L, E = x.shape # B x L x E

        q = self.q(y) # B x L x E
        kv = self.kv(x) # B x L x 2E
        q = q.reshape(B, L, self.n_heads, self.embed_dim // self.n_heads) # B x L x n_heads x (E // n_heads)
        q = q.permute(0, 2, 1, 3) # B x n_heads x L x (E // n_heads)
        kv = kv.reshape(B, L, self.n_heads, 2 * self.embed_dim // self.n_heads) # B x L x n_heads x (2E // n_heads)
        kv = kv.permute(0, 2, 1, 3) # B x n_heads x L x (2E // n_heads)
        k, v = kv.chunk(2, dim=-1) # B x n_heads x L x (E // n_heads)
        context_v, _ = scaled_dot_product_attention(q, k, v, mask) # B x n_heads x L x (E // n_heads)
        context_v_concat = context_v.permute(0, 2, 1, 3) # B x L x n_heads x (E // n_heads)
        context_v_concat = context_v_concat.reshape(B, L, E) # B x L x E
        context_v_concat = self.linear(context_v_concat) # B x L x E
        return context_v_concat


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden, dropout):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_hidden, embed_dim)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, L, embed_dim, n_heads, dropout, ff_hidden):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        # self.MultiHeadAttention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.LayerNorm1 = LayerNormalization(embed_dim=embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, ff_hidden=ff_hidden, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.LayerNorm2 = LayerNormalization(embed_dim=embed_dim)


    def forward(self, x, self_attention_mask): # B x L x E
        x_residual = x
        # x, _ = self.MultiHeadAttention(x, x, x) # B x L x E
        x = self.MultiHeadAttention(x, self_attention_mask) # B x L x E
        x = self.dropout1(x)
        x = x + x_residual
        x = self.LayerNorm1(x) # B x L x E

        x_residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + x_residual
        x = self.LayerNorm2(x)
        return x


class SentenceEmbedding(nn.Module):
    # language_to_index -> map token to index {token: index}
    def __init__(self, max_sequence_length, embed_dim, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, embed_dim) # B x L -> B x L x E
        self.language_to_index = language_to_index 
        self.position_encoder = PositionalEncoding(embed_dim, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return sentence_word_indicies # list
        
        tokenized = []
        for sentence in batch:
            tokenized.append(tokenize(sentence, start_token, end_token))
        tokenized = torch.tensor(tokenized) # B x L
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token) # B x L 
        x = self.embedding(x) # B x L x E
        pos = self.position_encoder().to(get_device()) # L x E
        # pos = self.position_encoder() # L x E
        x = self.dropout(x + pos)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, L, embed_dim, n_heads, dropout, ff_hidden, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(L, embed_dim, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.EncoderLayers = nn.ModuleList(
            [EncoderLayer(L=L, embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, ff_hidden=ff_hidden) for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token): # B x L
        x = self.sentence_embedding(x, start_token, end_token) # B x L x E
        for layer in self.EncoderLayers:
            x = layer(x, self_attention_mask) # B x L x E
        return x

class DecoderLayer(nn.Module):
    def __init__(self, L, embed_dim, n_heads, dropout, ff_hidden):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.LayerNorm1 = LayerNormalization(embed_dim=embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, ff_hidden=ff_hidden, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.LayerNorm2 = LayerNormalization(embed_dim=embed_dim)
        self.EncoderDecoderAttention = EncoderDecoderAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.dropout3 = nn.Dropout(dropout)
        self.LayerNorm3 = LayerNormalization(embed_dim=embed_dim)

    def forward(self, x, y, decoder_mask, cross_attention_mask): # B x L x E
        y_residual = y
        y = self.MultiHeadAttention(y, decoder_mask) # B x L x E
        y = self.dropout1(y)
        y = y + y_residual
        y = self.LayerNorm1(y) # B x L x E

        y_residual = y
        y = self.EncoderDecoderAttention(x, y, mask=cross_attention_mask) # B x L x E
        y = self.dropout2(y)
        y = y + y_residual
        y = self.LayerNorm2(y) # B x L x E

        y_residual = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = y + y_residual
        y = self.LayerNorm3(y) # B x L x E
        return y

class Decoder(nn.Module):
    def __init__(self, num_layers, L, embed_dim, n_heads, dropout, ff_hidden, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(L, embed_dim, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.DecoderLayers = nn.ModuleList(
            [DecoderLayer(L=L, embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, ff_hidden=ff_hidden) for _ in range(num_layers)])

    def forward(self, x, y, decoder_mask, cross_attention_mask, start_token, end_token): # X: B x L x E y: B x L
        y = self.sentence_embedding(y, start_token, end_token) # B x L x E
        for layer in self.DecoderLayers:
            y = layer(x, y, decoder_mask, cross_attention_mask) # B x L x E
        return y


class Transformer(nn.Module):
    def __init__(self, num_layers, L, embed_dim, n_heads, dropout, ff_hidden, english_to_index, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(kannada_to_index)
        self.encoder = Encoder(num_layers, L, embed_dim, n_heads, dropout, ff_hidden, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(num_layers, L, embed_dim, n_heads, dropout, ff_hidden, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(embed_dim, self.vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(
        self,
        x,
        y,
        encoder_mask=None,
        decoder_mask=None,
        cross_attention_mask=None,
        enc_start_token=False,
        enc_end_token=False,
        dec_start_token=False,
        dec_end_token=False,
    ):
        x = self.encoder(x, encoder_mask, enc_start_token, enc_end_token) # B x L x E
        y = self.decoder(x, y, decoder_mask, cross_attention_mask, dec_start_token, dec_end_token) # B x L x E
        y = self.linear(y) # B x L x vocab_size
        return y

