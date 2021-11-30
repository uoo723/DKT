import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        if args.interaction_type in [0, 2]:
            interaction_size = 2
        elif args.interaction_type == 3:
            interaction_size = 2 * args.n_tag
        else:
            interaction_size = 2 * args.n_questions

        self.embedding_interaction = nn.Embedding(
            interaction_size + 1, self.hidden_dim // 3, padding_idx=0
        )
        self.embedding_test = nn.Embedding(
            self.args.n_test + 1, self.hidden_dim // 3, padding_idx=0
        )
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3, padding_idx=0
        )
        self.embedding_tag = nn.Embedding(
            self.args.n_tag + 1, self.hidden_dim // 3, padding_idx=0
        )

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)  # (b, s, h)

        out = self.fc(out)  # (b, s, 1)
        preds = self.activation(out).view(batch_size, -1)  # (b, s)

        return preds
