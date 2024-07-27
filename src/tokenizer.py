import argparse
import datasets
import tokenizers
import utils
import huggingface_hub
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", metavar="REPOSITORY", help="Hugginface repository of the dataset to be used", default="gvlassis/shakespearefirstfolio")
parser.add_argument("--name", metavar="STRING", help="Name of the dataset config", default=None)
parser.add_argument("--split", metavar="STRING", default=None)
parser.add_argument("--vocab_size", metavar="INT", type=int, default=32000)
parser.add_argument("--batch_size", metavar="INT", type=int, default=1024)
args=parser.parse_args()

dataset = datasets.load_dataset(args.dataset, name=args.name, split=args.split, trust_remote_code=True)

tokenizer = tokenizers.Tokenizer(model=tokenizers.models.BPE())

tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
tokenizer.decoder = tokenizers.decoders.ByteLevel()

# Initial vocabulary
tokenizer.add_tokens([chr(i) for i in range(256)])
tokenizer.train_from_iterator(utils.dataset_iterator(dataset, args.batch_size), tokenizers.trainers.BpeTrainer(vocab_size=args.vocab_size), len(dataset))

print(tokenizer.get_vocab())

transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer).push_to_hub("%s_%s_%d" % (args.dataset.split("/")[1], args.split, args.vocab_size))
