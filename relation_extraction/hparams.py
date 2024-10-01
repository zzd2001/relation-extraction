import os
import argparse

# here = os.path.dirname(os.path.abspath(__file__))
# default_pretrained_model_path = os.path.join(here, '../pretrained_models/bert-base-chinese')
# default_train_file = os.path.join(here, '../datasets/train_small.jsonl')
# default_validation_file = os.path.join(here, '../datasets/val_small.jsonl')
# default_output_dir = os.path.join(here, '../saved_models')
# default_log_dir = os.path.join(default_output_dir, 'runs')
# default_tagset_file = os.path.join(here, '../datasets/relation.txt')
# default_model_file = os.path.join(default_output_dir, 'model.bin')
# default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')
# here = os.path.dirname(os.path.abspath(__file__))
default_pretrained_model_path = '/root/relation-extraction/pretrained_models/bert-base-chinese'
default_train_file = '/root/relation-extraction/datasets/historyRE_train.jsonl'
default_validation_file = '/root/relation-extraction/datasets/historyRE_val.jsonl'
default_test_file = '/root/relation-extraction/datasets/historyRE_test.jsonl'
default_output_dir = '/root/relation-extraction/saved_models'
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_file = '/root/relation-extraction/datasets/relation.txt'
default_model_file = os.path.join(default_output_dir, 'bert-base-chinese-model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'bert-base-chinese-checkpoint.json')
# ####################################
model_type = 'bert'
# ####################################
parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--validation_file", type=str, default=default_validation_file)
parser.add_argument("--test_file", type=str, default=default_test_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--tagset_file", type=str, default=default_tagset_file)
parser.add_argument("--model_file", type=str, default=default_model_file)  # 保存在验证集上表现最好的模型的权重
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')
parser.add_argument('--model_type', type=str, default=model_type)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--validation_batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)

hparams = parser.parse_args()
