import torch
import torch.nn as nn
import torch.nn.functional as F 

from utils import activation_getter

class CosRec(nn.Module):
    '''
    A 2D CNN for sequential Recommendation.

    Args:
        num_users: number of users.
        num_items: number of items.
        seq_len: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings.
        block_num: number of cnn blocks.
        block_dim: the dimensions for each block. len(block_dim)==block_num
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        ac_fc: type of activation functions.
        drop_prob: dropout ratio.
    '''
    def __init__(self, num_users, num_items, seq_len, embed_dim, block_num, block_dim, fc_dim, ac_fc, drop_prob):
        super(CosRec, self).__init__()
        assert len(block_dim) == block_num
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.cnnout_dim = block_dim[-1]
        self.fc_dim = fc_dim

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

        ### dropout and fc layer
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(self.cnnout_dim, self.fc_dim)
        self.ac_fc = activation_getter[ac_fc]

        ### build cnnBlock
        self.block_num = block_num
        block_dim.insert(0, 2*embed_dim)
        self.cnnBlock = [0] * block_num
        for i in range(block_num):
            self.cnnBlock[i] = cnnBlock(block_dim[i], block_dim[i+1])
        self.cnnBlock = nn.ModuleList(self.cnnBlock)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        '''
        Args:
            seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
                a batch of sequence
            user_var: torch.LongTensor with size [batch_size]
                a batch of user
            item_var: torch.LongTensor with size [batch_size]
                a batch of items
            for_pred: boolean, optional
                Train or Prediction. Set to True when evaluation.
        '''
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
        out = all_embed.permute(0, 3, 1, 2)

        # 2D CNN
        for i in range(self.block_num):
            out = self.cnnBlock[i](out)
        out = self.avgpool(out).reshape(mb, self.cnnout_dim)
        out = out.squeeze(-1).squeeze(-1)

        # apply fc and dropout
        out = self.ac_fc(self.fc1(out))
        out = self.dropout(out)

        x = torch.cat([out, user_emb.squeeze(1)], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        if for_pred:
            w2 = w2.squeeze() # (b,6,100)
            b2 = b2.squeeze() # (b,6)
            out = (x * w2).sum(1) + b2
        else:
            out = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze() # (b,6)

        return out

class cnnBlock(nn.Module):
    '''
    CNN block with two layers.
    
    For this illustrative model, we don't change stride and padding. 
    But to design deeper models, some modifications might be needed.
    '''
    def __init__(self, input_dim, output_dim, stride=1, padding=0):
        super(cnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

