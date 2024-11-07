import argparse
import os
import torch
import models.utils_models
import utils
import transformers
import numpy
import sklearn.metrics

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522)", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=16)
parser.add_argument("--context", type=int, default=1024)

parser.add_argument("--classes", help="Number of classes (e.g. animals, professions, colors) of words", type=int, default=3)
parser.add_argument("--string", help="The string to be visualized", default="Some animals: bear, cat, crab, crow, deer, dog, duck, eagle, elephant, fox, frog, horse, lion, monkey, rabbit, shark, snake, tiger, turtle, wolf. Some professions: actor, architect, artist, baker, biologist, chef, chemist, dentist, doctor, engineer, farmer, firefighter, journalist, lawyer, nurse, pilot, surgeon, teacher, waiter, writer. Some colors: black, blue, brown, gray, green, pink, purple, red, white, yellow.")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gpt2")

parser.add_argument("--balanced", help="Normalize to correct class imbalances", type=utils.str_to_bool, default=False)
parser.add_argument("--print_tokens", help="Print string tokens for debugging", type=utils.str_to_bool, default=False)
args=parser.parse_args()

similarity_path = args.PATH.split(".")[0]+"_similarity.dat"

device = "cuda:0"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.scale_type, args.ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, args.context, False, True)
model.load_state_dict(torch.load(args.PATH, weights_only=True))
model = model.to(device)

ids = args.tokenizer.encode(args.string).ids

animals = ["Ġbear", "Ġcat", "Ġcrab", "Ġcrow", "Ġdeer", "Ġdog", "Ġduck", "Ġeagle", "Ġelephant", "Ġfox", "Ġfrog", "Ġhorse", "Ġlion", "Ġmonkey", "Ġrabbit", "Ġshark", "Ġsnake", "Ġtiger", "Ġturtle", "Ġwolf"]
animals_ids = [args.tokenizer.get_vocab()[animal] for animal in animals]
animals_indices = [ids.index(animal_id) for animal_id in animals_ids]

professions = ["Ġactor", "Ġarchitect", "Ġartist", "Ġbaker", "Ġbiologist", "Ġchef", "Ġchemist", "Ġdentist", "Ġdoctor", "Ġengineer", "Ġfarmer", "Ġfirefighter", "Ġjournalist", "Ġlawyer", "Ġnurse", "Ġpilot", "Ġsurgeon", "Ġteacher", "Ġwaiter", "Ġwriter"]
professions_ids = [args.tokenizer.get_vocab()[profession] for profession in professions]
professions_indices = [ids.index(profession_id) for profession_id in professions_ids]

colors = ["Ġblack", "Ġblue", "Ġbrown", "Ġgray", "Ġgreen", "Ġpink", "Ġpurple", "Ġred", "Ġwhite", "Ġyellow"]
colors_ids = [args.tokenizer.get_vocab()[color] for color in colors]
colors_indices = [ids.index(color_id) for color_id in colors_ids]

if args.print_tokens:
    print(f"\x1b[1m{len(ids)} tokens\x1b[0m")

    for token in ids:
        if token in animals_ids:
            print("\x1b[38;2;27;158;119;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
        elif token in professions_ids:
            print("\x1b[38;2;217;95;2;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
        elif token in colors_ids:
            print("\x1b[38;2;117;112;179;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
        else:
            print("%16.16s: %6.6s" % (args.tokenizer.id_to_token(token), token))

X = torch.tensor(ids)

model.eval()
with torch.no_grad():
    embeddings = model.get_embeddings( X.to(device) )

print("\x1b[1m%6.6s %6.6s %6.6s %6.6s %6.6s\x1b[0m" % ("block", "intra", "inter", "ratio", "sil"))
for block in range(model.num_blocks+1):
    animals_embeddings = [embeddings[block,animal_index,:] for animal_index in animals_indices]
    professions_embeddings = [embeddings[block,profession_index,:] for profession_index in professions_indices]
    colors_embeddings = [embeddings[block,color_index,:] for color_index in colors_indices]
    X = numpy.array(animals_embeddings+professions_embeddings+colors_embeddings)
    Y = [0]*len(animals) + [1]*len(professions) + [2]*len(colors)
    intra = utils.intra(X,Y,args.balanced)
    inter = utils.inter(X,Y,args.balanced)
    ratio = inter/intra
    sil = sklearn.metrics.silhouette_score(X, Y, metric="cosine")
    print("%6.6s %6.6s %6.6s %6.6s %6.6s" % (block, "%.2f" % intra, "%.2f" % inter, "%.2f" % ratio, "%.2f" % sil))

similarity_header = models.transformer.get_similarity_header(model)
with open(similarity_path,"w") as file:
    file.write(f"x y token1 token2 {similarity_header}\n")

# How similar token1 (x in matrix plot) is to token2 (y in matrix plot)
y=0
for j, token2 in enumerate(ids):
    if j not in animals_indices+professions_indices+colors_indices:
        continue

    x=0
    for i, token1 in enumerate(ids):
        if i not in animals_indices+professions_indices+colors_indices:
            continue
        similarity = models.transformer.get_similarity(embeddings[:, i, :], embeddings[:, j, :])
        with open(similarity_path,"a") as file:
            file.write("%d %d %s %s %s\n" % (x, y, args.tokenizer.id_to_token(token1), args.tokenizer.id_to_token(token2), similarity))
        x+=1
        
    with open(similarity_path,"a") as file:
            file.write("\n")
    y+=1
