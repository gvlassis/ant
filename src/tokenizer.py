import argparse
import datasets
import tokenizers
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Where tokenizer will be saved")
parser.add_argument("--dataset", metavar="REPOSITORY", help="Hugginface repository of the dataset to be used", default="gvlassis/ancient_greek_theatre")
parser.add_argument("--name", metavar="STRING", help="Name of the dataset config", default=None)
parser.add_argument("--split", metavar="STRING", default=None)
parser.add_argument("--vocab_size", metavar="INT", type=int, default=32000)
parser.add_argument("--special_tokens", metavar="TOK1,...", type=utils.str_to_list, default=["[CLS]","[BOS]","[UNK]","[MASK]","[EOS]","[SEP]","[EOT]","[PAD]"])
parser.add_argument("--batch_size", metavar="INT", type=int, default=1024)
args=parser.parse_args()

dataset = datasets.load_dataset(args.dataset, name=args.name, split=args.split, trust_remote_code=True)

tokenizer = tokenizers.Tokenizer(model=tokenizers.models.BPE())

tokenizer.normalizer = tokenizers.normalizers.NFKD()
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = tokenizers.decoders.ByteLevel()

tokenizer.add_special_tokens(args.special_tokens)
tokenizer.train_from_iterator(utils.dataset_iterator(dataset, args.batch_size), tokenizers.trainers.BpeTrainer(vocab_size=args.vocab_size-len(args.special_tokens)), len(dataset))

tokenizer.save(args.PATH)
