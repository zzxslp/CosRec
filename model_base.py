import torch
import torch.nn as nn
import torch.nn.functional as F 

class CosRec_base(nn.Module):
    '''
    A baseline model using MLP for ablation studies.
    '''
    def __init__(self, num_users, num_items, seq_len, embed_dim):
        super(CosRec_base, self).__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.fc_dim = 100

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, self.fc_dim+embed_dim)
        self.b2 = nn.Embedding(num_items, 1)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        # other units
        self.g_fc1 = nn.Linear(2*embed_dim, self.fc_dim)
        self.g_fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        mb = seq_var.shape[0]
        item_embs = self.item_embeddings(seq_var) # (b, L, embed)(b, 5, 50)
        user_emb = self.user_embeddings(user_var) # (b, 1, embed)

        # add user embedding everywhere
        usr = user_emb.repeat(1, self.seq_len, 1) # (b, 5, embed)
        usr = torch.unsqueeze(usr, 2) # (b, 5, 1, embed)

        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(item_embs, 1) # (b, 1, 5, embed)
        item_i = item_i.repeat(1, self.seq_len, 1, 1) # (b, 5, 5, embed)
        item_j = torch.unsqueeze(item_embs, 2) # (b, 5, 1, embed)
        item_j = item_j.repeat(1, 1, self.seq_len, 1) # (b, 5, 5, embed)

        all_embed = torch.cat([item_i, item_j], 3) # (b, 5, 5, 2*embed)

        x_ = all_embed.view(-1, 2*self.embed_dim)
        x_ = F.relu(self.g_fc1(x_))
        x_ = F.relu(self.g_fc2(x_))
        x_ = self.dropout(x_)

        x_g = x_.view(mb, -1, self.fc_dim)
        x = x_g.sum(1)
        x = torch.cat([x, user_emb.squeeze(1)], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        if for_pred:
            w2 = w2.squeeze() # (b,6,100)
            b2 = b2.squeeze() # (b,6)
            out = (x * w2).sum(1) + b2
        else:
            out = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze() # (b,6)

        return out
