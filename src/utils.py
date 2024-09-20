import torch
import os
import re
import time
import unicodedata
import copy
import plotext
import warnings
warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")

def get_files(root):
    root = os.path.abspath(root)

    files_abs = []
    for directory, _, files_rel in os.walk(root):
        for file in files_rel:
            files_abs.append(directory+"/"+file)

    return files_abs

def match_list(_list, regex, group):
    list_matched = set()
    
    for elem in _list:
        match = re.search(regex, elem)
        if match:
            list_matched.add(match.group(group))

    return list(list_matched)

def get_subdir(path):
    return [child for child in os.listdir(path) if os.path.isdir(path+"/"+child)]

def get_subdat(path):
    return [child[:-4] for child in os.listdir(path) if child[-4:]==".dat"]

def numel(elem):
    if type(elem) == list:
        return sum(numel(subelem) for subelem in elem)
    else:
        return 1

def str_to_bool(string):
    if string == "True":
        boolean = True
    elif string == "False":
        boolean = False

    return boolean

# Time logging causes negligible performance impact (~3%)
def get_sync_time(device):
    torch.cuda.synchronize(device)

    return round(time.time()*1e6)

# Based on https://github.com/gvlassis/bashrc_utils/blob/main/src/utils.sh
def us_to_human_friendly(μs):
    ms = μs//1000

    s = ms//1000
    
    rem_s = s%60
    m = s//60
    
    rem_m = m%60
    h = m//60

    rem_h = h%24
    d = h//24

    if d > 0:
        human_friendly = "%dd%dh" % (d,rem_h)
    elif h > 0:
        human_friendly = "%dh%dm" % (h,rem_m)
    elif m > 0:
        human_friendly = "%dm%ds" % (m,rem_s)
    elif s > 0:
        # s=AB, ms=ABCDE
        if s > 10:
            human_friendly = "%d.%ss" % (s,str(ms)[2])
        # s=A, ms=ABCD
        else:
            human_friendly = "%d.%ss" % (s,str(ms)[1:3])
    elif ms > 0:
        # ms=ABC
        if ms > 100:
            human_friendly = "%dms" % ms
        # ms=AB, μs=ABCDE
        elif ms > 10:
            human_friendly = "%d.%sms" % (ms,str(μs)[2])
        # ms=A, μs=ABCD
        else:
            human_friendly = "%d.%sms" % (ms,str(μs)[1:3])
    else:
        human_friendly = "%dμs" % μs

    return human_friendly

def topp(A, P):
    A_values, A_indices = torch.sort(A, descending=True)

    A_cumsum = torch.cumsum(A_values, dim=0)

    indices = A_indices[A_cumsum < P]

    indices = A_indices[:len(indices)+1]

    return A[indices], indices

def intersection(A, B):
    uniques, counts = torch.cat((A, B)).unique(return_counts=True)
    
    return uniques[counts>1]

def generate_text(starting_string, tokenizer, unk_id, eot_id, model, context=128, max_tokens=128, T=1, K=50, P=0.95):
    string = starting_string

    device = next(model.parameters()).device

    ids = tokenizer.encode(starting_string).ids

    while ( ids[-1] != eot_id ) and ( len(ids) < max_tokens ):

        X = torch.tensor(ids[-context:])

        model.eval()
        with torch.no_grad():
            Y = model( X.to(device) )
        
        Y[-1][unk_id] = -float("inf")

        Y_ = torch.nn.functional.softmax(Y[-1]/T, dim=0)
        
        topk_indices = torch.topk(Y_, K).indices
        topp_indices = topp(Y_, P)[1]
        indices = intersection(topk_indices, topp_indices)
        
        Y__ = torch.zeros_like(Y_)
        Y__[indices] = Y_[indices]

        ids = ids + [torch.multinomial(Y__, num_samples=1).item()]
    
    string = tokenizer.decode(ids)

    string = unicodedata.normalize("NFKC", string)
    
    print(string)

def print_schedule(train_batches, scheduler):
    scheduler = copy.deepcopy(scheduler)
    
    lrs = []
    for train_batch in range(train_batches):
        # optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plotext.plot_size(width=plotext.terminal_width()/2, height=plotext.terminal_height()/4)
    plotext.theme("pro")
    plotext.xlabel("train_batch")
    plotext.xaxes(lower=True, upper=False)
    plotext.yaxes(left=True, right=False)
    plotext.xticks([0,train_batches*0.25,train_batches*0.5,train_batches*0.75,train_batches])
    
    plotext.plot(range(train_batches), lrs, marker="braille", label="lr")
    plotext.show()
    plotext.clear_figure()
