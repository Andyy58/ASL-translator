import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ASLSequentialProcessor(nn.Module):
    def __init__(
        self,
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=100,
        dropout=0.4,
    ):
        super(ASLSequentialProcessor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):

        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (h_n, c_n) = self.lstm(packed_input)

        final_hidden = h_n[-1]

        final_hidden = self.dropout(final_hidden)
        out = self.fc(final_hidden)

        return out
