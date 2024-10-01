import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel

here = os.path.dirname(os.path.abspath(__file__))


class SentenceRE(nn.Module):

    def __init__(self, hparams):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'roberta-base'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        self.roberta_model = AutoModel.from_pretrained(self.pretrained_model_path)

        # 扩展模型词嵌入以适应新标记
        new_tokenizer_file = os.path.join(self.pretrained_model_path, 'roberta_tokenizer_with_special_tokens')
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_file)
        self.roberta_model.resize_token_embeddings(len(new_tokenizer))

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3)
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)

    def forward(self, token_ids, attention_mask, e1_mask, e2_mask):
        # RoBERTa不使用token_type_ids
        sequence_output, pooled_output = self.roberta_model(input_ids=token_ids, attention_mask=attention_mask, return_dict=False)

        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

# import os
# import torch
# import torch.nn as nn
# from transformers import BertModel, RobertaModel

# here = os.path.dirname(os.path.abspath(__file__))

# class SentenceRE(nn.Module):
#     def __init__(self, hparams):
#         super(SentenceRE, self).__init__()

#         self.model_type = hparams.model_type  # 'bert' or 'roberta'
#         self.pretrained_model_path = hparams.pretrained_model_path
#         self.embedding_dim = hparams.embedding_dim
#         self.dropout = hparams.dropout
#         self.tagset_size = hparams.tagset_size

#         if self.model_type == 'bert':
#             self.model = BertModel.from_pretrained(self.pretrained_model_path)
#         elif self.model_type == 'roberta':
#             self.model = RobertaModel.from_pretrained(self.pretrained_model_path)
#         else:
#             raise ValueError("Unsupported model type. Choose 'bert' or 'roberta'.")

#         self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.drop = nn.Dropout(self.dropout)
#         self.activation = nn.Tanh()
#         self.norm = nn.LayerNorm(self.embedding_dim * 3)
#         self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)

#     def forward(self, token_ids, attention_mask, e1_mask, e2_mask, token_type_ids=None):
#         if self.model_type == 'bert':
#             sequence_output, pooled_output = self.model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
#         elif self.model_type == 'roberta':
#             print(token_ids)
#             print(token_ids.shape)
#             print(attention_mask)
#             print(attention_mask.shape)
#             sequence_output = self.model(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state
#             pooled_output = sequence_output[:, 0, :]  # RoBERTa的[CLS]位置

#         # 每个实体的所有token向量的平均值
#         e1_h = self.entity_average(sequence_output, e1_mask)
#         e2_h = self.entity_average(sequence_output, e2_mask)
#         e1_h = self.activation(self.dense(e1_h))
#         e2_h = self.activation(self.dense(e2_h))

#         # [cls] + 实体1 + 实体2
#         concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
#         concat_h = self.norm(concat_h)
#         logits = self.hidden2tag(self.drop(concat_h))

#         return logits

#     @staticmethod
#     def entity_average(hidden_output, e_mask):
#         e_mask_unsqueeze = e_mask.unsqueeze(1)
#         length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

#         sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
#         avg_vector = sum_vector.float() / length_tensor.float()
#         return avg_vector


# import os
# import logging
# import torch
# import torch.nn as nn

# from transformers import BertTokenizer, BertModel

# here = os.path.dirname(os.path.abspath(__file__))


# class SentenceRE(nn.Module):

#     def __init__(self, hparams):
#         super(SentenceRE, self).__init__()

#         self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
#         self.embedding_dim = hparams.embedding_dim
#         self.dropout = hparams.dropout
#         self.tagset_size = hparams.tagset_size

#         self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)

#         self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.drop = nn.Dropout(self.dropout)
#         self.activation = nn.Tanh()
#         self.norm = nn.LayerNorm(self.embedding_dim * 3)
#         self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)

#     def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
#         sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)

#         # 每个实体的所有token向量的平均值
#         e1_h = self.entity_average(sequence_output, e1_mask)
#         e2_h = self.entity_average(sequence_output, e2_mask)
#         e1_h = self.activation(self.dense(e1_h))
#         e2_h = self.activation(self.dense(e2_h))

#         # [cls] + 实体1 + 实体2
#         concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
#         concat_h = self.norm(concat_h)
#         logits = self.hidden2tag(self.drop(concat_h))

#         return logits

#     @staticmethod
#     def entity_average(hidden_output, e_mask):
#         """
#         Average the entity hidden state vectors (H_i ~ H_j)
#         :param hidden_output: [batch_size, j-i+1, dim]
#         :param e_mask: [batch_size, max_seq_len]
#                 e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
#         :return: [batch_size, dim]
#         """
#         e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
#         length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

#         sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
#         avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
#         return avg_vector
