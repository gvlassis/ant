import argparse
import os
import torch
import models.utils_models
import transformers
import utils
import numpy
import sklearn.random_projection
import sklearn.decomposition
import sklearn.manifold
import umap

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522)", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=16)
parser.add_argument("--context", type=int, default=1024)

parser.add_argument("--string", help="The string to be visualized", default="Some animals: cat, dog, tiger, horse, bat, eagle. Some professions: chef, vet, doctor, pilot, waiter, nurse. Some colors: brown, green, blue, red, yellow, purple.")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gpt2")

parser.add_argument("--blocks_interval", help="Every how many transformer blocks to check", type=int, default=1)
args=parser.parse_args()

clustering_path = args.PATH.split(".")[0]+"_clustering.dat"

device = "cuda:0"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.scale_type, args.ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, args.context, False, True)
model.load_state_dict(torch.load(args.PATH, weights_only=True))
model = model.to(device)

ids = args.tokenizer.encode(args.string).ids
print(f"\x1b[1m{len(ids)} tokens\x1b[0m")

animals = ["Ġcat", "Ġdog", "Ġtiger", "Ġhorse", "Ġbat", "Ġeagle"]
animals_ids = [args.tokenizer.get_vocab()[animal] for animal in animals]
animals_indices = [ids.index(animal_id) for animal_id in animals_ids]

professions = ["Ġchef", "Ġvet", "Ġdoctor", "Ġpilot", "Ġwaiter", "Ġnurse"]
professions_ids = [args.tokenizer.get_vocab()[profession] for profession in professions]
professions_indices = [ids.index(profession_id) for profession_id in professions_ids]

colors = ["Ġbrown", "Ġgreen", "Ġblue", "Ġred", "Ġyellow", "Ġpurple"]
colors_ids = [args.tokenizer.get_vocab()[color] for color in colors]
colors_indices = [ids.index(color_id) for color_id in colors_ids]

for token in ids:
    if token in animals_ids:
        print("\x1b[31;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
    elif token in professions_ids:
        print("\x1b[32;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
    elif token in colors_ids:
        print("\x1b[34;1m%16.16s: %6.6s\x1b[0m" % (args.tokenizer.id_to_token(token), token))
    else:
        print("%16.16s: %6.6s" % (args.tokenizer.id_to_token(token), token))

X = torch.tensor(ids)

model.eval()
with torch.no_grad():
    embeddings = model.get_embeddings( X.to(device) )

animals_random = numpy.empty((len(animals), model.num_blocks+1, 2))
professions_random = numpy.empty((len(professions), model.num_blocks+1, 2))
colors_random = numpy.empty((len(colors), model.num_blocks+1, 2))

animals_pca = numpy.empty((len(animals), model.num_blocks+1, 2))
professions_pca = numpy.empty((len(professions), model.num_blocks+1, 2))
colors_pca = numpy.empty((len(colors), model.num_blocks+1, 2))

animals_mds = numpy.empty((len(animals), model.num_blocks+1, 2))
professions_mds = numpy.empty((len(professions), model.num_blocks+1, 2))
colors_mds = numpy.empty((len(colors), model.num_blocks+1, 2))

animals_tsne = numpy.empty((len(animals), model.num_blocks+1, 2))
professions_tsne = numpy.empty((len(professions), model.num_blocks+1, 2))
colors_tsne = numpy.empty((len(colors), model.num_blocks+1, 2))

animals_umap = numpy.empty((len(animals), model.num_blocks+1, 2))
professions_umap = numpy.empty((len(professions), model.num_blocks+1, 2))
colors_umap = numpy.empty((len(colors), model.num_blocks+1, 2))


# embedding
animals_embeddings = [embeddings[0,animal_index,:] for animal_index in animals_indices]
professions_embeddings = [embeddings[0,profession_index,:] for profession_index in professions_indices]
colors_embeddings = [embeddings[0,color_index,:] for color_index in colors_indices]

random = sklearn.random_projection.GaussianRandomProjection(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
animals_random[:,0,:] = random[:len(animals)]
professions_random[:,0,:] = random[len(animals):len(animals)+len(professions)]
colors_random[:,0,:] = random[len(animals)+len(professions):]

pca = sklearn.decomposition.PCA(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
animals_pca[:,0,:] = pca[:len(animals)]
professions_pca[:,0,:] = pca[len(animals):len(animals)+len(professions)]
colors_pca[:,0,:] = pca[len(animals)+len(professions):]

mds = sklearn.manifold.MDS(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
animals_mds[:,0,:] = mds[:len(animals)]
professions_mds[:,0,:] = mds[len(animals):len(animals)+len(professions)]
colors_mds[:,0,:] = mds[len(animals)+len(professions):]

tsne = sklearn.manifold.TSNE(n_components=2, perplexity=6).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
animals_tsne[:,0,:] = tsne[:len(animals)]
professions_tsne[:,0,:] = tsne[len(animals):len(animals)+len(professions)]
colors_tsne[:,0,:] = tsne[len(animals)+len(professions):]

umap_ = umap.UMAP(n_components=2, n_neighbors=6).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
animals_umap[:,0,:] = umap_[:len(animals)]
professions_umap[:,0,:] = umap_[len(animals):len(animals)+len(professions)]
colors_umap[:,0,:] = umap_[len(animals)+len(professions):]


# Transformer blocks
for block in range(model.num_blocks):
    if block % args.blocks_interval == 0:
        animals_embeddings = [embeddings[block+1,animal_index,:] for animal_index in animals_indices]
        professions_embeddings = [embeddings[block+1,profession_index,:] for profession_index in professions_indices]
        colors_embeddings = [embeddings[block+1,color_index,:] for color_index in colors_indices]

        random = sklearn.random_projection.GaussianRandomProjection(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
        animals_random[:,block+1,:] = random[:len(animals)]
        professions_random[:,block+1,:] = random[len(animals):len(animals)+len(professions)]
        colors_random[:,block+1,:] = random[len(animals)+len(professions):]

        pca = sklearn.decomposition.PCA(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
        animals_pca[:,block+1,:] = pca[:len(animals)]
        professions_pca[:,block+1,:] = pca[len(animals):len(animals)+len(professions)]
        colors_pca[:,block+1,:] = pca[len(animals)+len(professions):]

        mds = sklearn.manifold.MDS(n_components=2).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
        animals_mds[:,block+1,:] = mds[:len(animals)]
        professions_mds[:,block+1,:] = mds[len(animals):len(animals)+len(professions)]
        colors_mds[:,block+1,:] = mds[len(animals)+len(professions):]

        tsne = sklearn.manifold.TSNE(n_components=2, perplexity=6).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
        animals_tsne[:,block+1,:] = tsne[:len(animals)]
        professions_tsne[:,block+1,:] = tsne[len(animals):len(animals)+len(professions)]
        colors_tsne[:,block+1,:] = tsne[len(animals)+len(professions):]

        umap_ = umap.UMAP(n_components=2, n_neighbors=6).fit_transform(numpy.array(animals_embeddings+professions_embeddings+colors_embeddings))
        animals_umap[:,block+1,:] = umap_[:len(animals)]
        professions_umap[:,block+1,:] = umap_[len(animals):len(animals)+len(professions)]
        colors_umap[:,block+1,:] = umap_[len(animals)+len(professions):]

clustering_header = models.transformer.get_clustering_header(model, args.blocks_interval)
with open(clustering_path,"w") as file:
    file.write(f"token class {clustering_header}\n")

for i, animal in enumerate(animals):
    
    clustering = models.transformer.get_clustering(animals_random[i], animals_pca[i], animals_mds[i], animals_tsne[i], animals_umap[i], args.blocks_interval)

    with open(clustering_path,"a") as file:
        file.write(f"{animal} animals {clustering}\n")

for i, profession in enumerate(professions):
    
    clustering = models.transformer.get_clustering(professions_random[i], professions_pca[i], professions_mds[i], professions_tsne[i], professions_umap[i], args.blocks_interval)

    with open(clustering_path,"a") as file:
        file.write(f"{profession} professions {clustering}\n")

for i, color in enumerate(colors):
    
    clustering = models.transformer.get_clustering(colors_random[i], colors_pca[i], colors_mds[i], colors_tsne[i], colors_umap[i], args.blocks_interval)

    with open(clustering_path,"a") as file:
        file.write(f"{color} colors {clustering}\n")
