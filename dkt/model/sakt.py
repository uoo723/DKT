import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAKT(nn.Module):
    """Implementation of "A Self-Attentive model for Knowledge Tracing"
    https://arxiv.org/abs/1907.06837

    Ref: https://github.com/arshadshk/SAKT-pytorch
    """

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.seq_len = args.max_seq_len
        self.attn_direction = args.attn_direction
        self.interaction_type = args.interaction_type

        # interaction embeddings
        if args.interaction_type in [0, 2]:
            interaction_size = 2
        elif args.interaction_type == 3:
            interaction_size = 2 * args.n_tag
        else:
            interaction_size = 2 * args.n_questions

        self.embd_in = nn.Embedding(
            interaction_size + 1, args.hidden_dim, padding_idx=0
        )

        if self.interaction_type == 2:
            self.in_proj = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        else:
            self.register_parameter("in_proj", None)

        # excercise embeddings
        self.embd_ex = nn.Embedding(
            args.n_questions + 1, args.hidden_dim, padding_idx=0
        )

        # position embeddings
        self.embd_pos = nn.Embedding(args.max_seq_len, args.hidden_dim)

        self.linear = nn.ModuleList(
            nn.Linear(args.hidden_dim, args.hidden_dim) for _ in range(3)
        )
        self.attn = nn.MultiheadAttention(args.hidden_dim, args.n_heads, args.drop_out)
        self.ffn = nn.ModuleList(
            nn.Linear(args.hidden_dim, args.hidden_dim) for _ in range(2)
        )

        self.linear_out = nn.Linear(args.hidden_dim, 1)
        self.layer_norm1 = nn.LayerNorm(args.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(args.hidden_dim)
        self.dropout = nn.Dropout(args.drop_out)

    def _get_pos_idx(self, mask):
        # pos_idx = torch.zeros(mask.shape[0], self.seq_len, dtype=torch.int64)
        # for i, m in enumerate(mask):
        #     idx = m.nonzero(as_tuple=True)[0]
        #     pos_idx[i, idx] = torch.arange(1, len(idx) + 1)
        pos_idx = torch.arange(self.seq_len).unsqueeze(0)
        return pos_idx

    def forward(self, inputs):
        _, question, _, _, mask, interaction, _ = inputs

        # positional embedding
        pos_idx = self._get_pos_idx(mask)
        pos_in = self.embd_pos(pos_idx.to(self.device))

        # excercise embedding
        query_ex = self.embd_ex(question)

        # interaction embedding
        out_in = self.embd_in(interaction)

        if self.interaction_type == 2:
            out_in = self.in_proj(torch.cat([query_ex, out_in], dim=-1))

        out_in = out_in + pos_in

        # split the interaction embedding into v and k
        value_in = out_in
        key_in = out_in

        # Linear projection all the embeddings
        value_in = self.linear[0](value_in).permute(1, 0, 2)
        key_in = self.linear[1](key_in).permute(1, 0, 2)
        query_ex = self.linear[2](query_ex).permute(1, 0, 2)

        # pass through multi-head attention
        attn_mask = torch.from_numpy(
            np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype("bool")
        ).to(self.device)
        attn_out, _ = self.attn(
            query_ex,
            key_in,
            value_in,
            attn_mask=attn_mask if self.attn_direction == "uni" else None,
        )
        attn_out = query_ex + attn_out
        attn_out = self.layer_norm1(attn_out)
        attn_out = attn_out.permute(1, 0, 2)

        # FFN layers
        ffn_out = self.dropout(self.ffn[1](F.relu(self.ffn[0](attn_out))))
        ffn_out = self.layer_norm2(ffn_out + attn_out)

        # Sigmoid
        ffn_out = torch.sigmoid(self.linear_out(ffn_out).squeeze(-1))

        return ffn_out
