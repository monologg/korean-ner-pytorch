import torch
import torch.nn as nn
from torchcrf import CRF

from utils import get_labels


class CharCNN(nn.Module):
    def __init__(self,
                 max_word_len=10,
                 kernel_lst="2,3,4",
                 num_filters=32,
                 char_vocab_size=1000,
                 char_emb_dim=30,
                 final_char_dim=50):
        super(CharCNN, self).__init__()

        # Initialize character embedding
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.25, 0.25)

        kernel_lst = list(map(int, kernel_lst.split(",")))  # "2,3,4" -> [2, 3, 4]

        # Convolution for each kernel
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2),
                nn.Tanh(),  # As the paper mentioned
                nn.MaxPool1d(max_word_len - kernel_size + 1),
                nn.Dropout(0.25)  # As same as the original code implementation
            ) for kernel_size in kernel_lst
        ])

        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(kernel_lst), 100),
            nn.ReLU(),  # As same as the original code implementation
            nn.Dropout(0.25),
            nn.Linear(100, final_char_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, final_char_dim)
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_emb(x)  # (b, s, w, d)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)  # (b*s, w, d)
        x = x.transpose(2, 1)  # (b*s, d, w): Conv1d takes in (batch, dim, seq_len), but raw embedded is (batch, seq_len, dim)

        conv_lst = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_lst, dim=-1)  # (b*s, num_filters, len(kernel_lst))
        conv_concat = conv_concat.view(conv_concat.size(0), -1)  # (b*s, num_filters * len(kernel_lst))

        output = self.linear(conv_concat)  # (b*s, final_char_dim)
        output = output.view(batch_size, max_seq_len, -1)  # (b, s, final_char_dim)
        return output


class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self, args, pretrained_word_matrix):
        super(BiLSTM_CNN_CRF, self).__init__()
        self.args = args

        self.char_cnn = CharCNN(max_word_len=args.max_word_len,
                                kernel_lst=args.kernel_lst,
                                num_filters=args.num_filters,
                                char_vocab_size=args.char_vocab_size,
                                char_emb_dim=args.char_emb_dim,
                                final_char_dim=args.final_char_dim)

        if pretrained_word_matrix is not None:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_word_matrix)
        else:
            self.word_emb = nn.Embedding(args.word_vocab_size, args.word_emb_dim, padding_idx=0)
            nn.init.uniform_(self.word_emb.weight, -0.25, 0.25)

        self.bi_lstm = nn.LSTM(input_size=args.word_emb_dim + args.final_char_dim,
                               hidden_size=args.hidden_dim // 2,  # Bidirectional will double the hidden_size
                               bidirectional=True,
                               batch_first=True)

        self.output_linear = nn.Linear(args.hidden_dim, len(get_labels(args)))

        self.crf = CRF(num_tags=len(get_labels(args)), batch_first=True)

    def forward(self, word_ids, char_ids, mask, label_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return: (batch_size, max_seq_len, hidden_dim)
        """
        w_emb = self.word_emb(word_ids)
        c_emb = self.char_cnn(char_ids)

        w_c_emb = torch.cat([w_emb, c_emb], dim=-1)

        lstm_output, _ = self.bi_lstm(w_c_emb, None)

        output = self.output_linear(lstm_output)

        loss = 0
        if label_ids is not None:
            loss = self.crf(output, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, output
