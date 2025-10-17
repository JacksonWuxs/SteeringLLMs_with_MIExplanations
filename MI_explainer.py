import os
import gc
import sys
import random

import tqdm
import transformers as trf
import torch as tc
import numpy as np

from autoencoder import load_pretrained
from generator import Generator


random.seed(42)
trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
CACHE_DIR = "/data/Huggingface"


def load_c4_dataset(fpath):
    with open(fpath, encoding="utf8") as f:
        for row in f:
            yield row.split("\t")[-1].strip() 


class TunedLensTranslator(tc.nn.Module):
    def __init__(self, dim, norm, device="cuda"):
        tc.nn.Module.__init__(self)
        self._device = device
        self._normalize = (norm,)
        self._translator = tc.nn.Linear(dim, dim, bias=True, device=self._device)
        self._translator.weight.data.fill_(0.)
        self._translator.bias.data.fill_(0.)
        self.to(device)
    
    def forward(self, X):
        X = X.to(self._device).float()
        #X = self._translator(X) + X
        return self._normalize[0](X.bfloat16())

    def compute_loss(self, X, Y, E, T=2):
        if T != 1:
            Y, P = Y / 2, P / 2
        Y, P = Y.log_softmax(dim=-1), E(self(X)).log_softmax(-1)
        return (Y.exp() * (Y - P)).sum(-1).mean()

    def clip_grad(self, max_norm=1):
        tc.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    @classmethod
    def load_disk(cls, fpath, norm, device="cuda"):
        assert os.path.exists(fpath)
        weights = tc.load(fpath)
        model = cls(weights["weight"].shape[0], norm, device)
        model._translator.load_state_dict(weights)
        print("Translator is loaded from %s." % fpath)
        model = model.to(device)
        model.eval()
        return model
    
    def dump_disk(self, fpath):
        tc.save(self._translator.state_dict(), fpath)
        print("Translator is dumped at %s." % fpath)
        


class FIFOStack:
    def __init__(self, size=10):
        self._size = size
        self._items = []

    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        for item in self._items:
            yield item

    def append(self, item):
        if len(self._items) >= self._size:
            self._items.pop(0)
        self._items.append(item)

    def average(self):
        return np.mean(self._items)

    def count_trends(self, lags=3, eps=1e-5):
        last_item = None
        pos, eq, neg = 0, 0, 0
        for next_item in self._items[-lags-1:]:
            if last_item is not None:
                diff = next_item - last_item
                if diff > eps:
                    pos += 1
                elif diff < -eps:
                    neg += 1
                else:
                    eq += 1
            last_item = next_item 
        return pos, eq, neg



def normalize_word(word):
    #w = word.lower()
    w = word
    if len(w) > 3 and w[-3:] in ("ies", "ied"):
        return w[:-3] + 'y'
    if w.endswith("ves") and len(w) > 3:
        return w[:-3] + "f"
    if len(w) > 3 and w[-3:] in ("ing", "est"):
        return w[:-3]
    if len(w) > 2 and w[-2:] in {"es", "ed", "er", "ly", "'s"}:
        return w[:-2]
    if len(w) > 1 and w[-1] in ("s", ".", "'"):
        return w[:-1]
    return w



def load_wiki_words(fpath, minfreq=5.0, only_ascii=False, only_alnum=False):
    from collections import defaultdict
    words = defaultdict(int) 
    with open(fpath, encoding="utf8") as f:
        for row in f:
            try:
                if "\t" in row or " " in row:
                    word, freq = row.strip().split()
                else:
                    word, freq = row.strip(), minfreq + 1
                if only_ascii and not word.isascii():
                    continue
                if only_alnum and not word.isalnum():
                    continue
                if word.startswith("##"):
                    continue
                words[word] += float(freq)
            except:
                pass

    merged = defaultdict(int)
    skipped = set()
    for key, val in words.items():
        if key in skipped:
            continue
        normed = normalize_word(key)
        if normed in words:
            merged[normed] = val + words[normed]
            skipped.add(normed)

    tokens, freqs = [], []
    for pair in merged.items():
        if pair[1] >= minfreq:
            tokens.append(pair[0])
            freqs.append(pair[1] + 1)
    words = [_[0] for _ in words.items() if _[1] >= minfreq]
    print("Loading %d words from %s" % (len(words), fpath))
    return words


class MIExplainer(Generator):
    def __init__(self, model_name, device="cuda", dtype="bfloat16"):
        Generator.__init__(self, model_name, device)
        self._vocab = None
        self._translator = None

    @tc.no_grad()
    def build_vocab(self, words, batch_size=2048):
        self._words = words

        def encode_batch(ids, H):
            M, maxlen = [], max(map(len, ids))
            for _ in ids:
                M.append([1] * len(_) + [0] * (maxlen - len(_)))
                _.extend([self._tokenizer.eos_token_id] * (maxlen - len(_)))
            M = tc.tensor(M).float().unsqueeze(-1).cpu()
            ids = tc.tensor(ids).long()
            H.append((Eo[ids] * M).sum(axis=1) / (1e-9 + M.sum(axis=1)))
        
        Eo = self._out_emb.weight.cpu()
        H, batchE = [], []
        for word in tqdm.tqdm(self._words, desc="Preparing Vocab"):
            tokens = self._tokenizer.tokenize(" " + word)
            if tokens[0] in (u'Ġ', u'▁'):
                if tokens[1][0] in (u'Ġ', u'▁'):
                    tokens = tokens[1:]
            batchE.append(self._tokenizer.convert_tokens_to_ids(tokens))
            if len(batchE) == batch_size:
                encode_batch(batchE, H)
                batchE.clear()
        if len(batchE) > 0:
            encode_batch(batchE, H)
        self._vocab = tc.cat(H, axis=0)

    def build_translator(self, corpus, layer, batch_size, learn_rate, total_steps=2000, name='', T=1, eps=1e-4):
        dim = self._out_emb.weight.shape[-1]
        fpath = "outputs/weights/tunedlens_%s_layer%d.pth" % (name, layer)
        if os.path.exists(fpath):
            self._translator = TunedLensTranslator.load_disk(fpath, self._model.base_model.norm, self._device)
            return 
        self._translator = TunedLensTranslator(dim, self._model.base_model.norm, self._device)
        
        optim = tc.optim.AdamW(self._translator.parameters(), lr=learn_rate, betas=(0.95, 0.9995), weight_decay=0.01)
        
        history = FIFOStack(size=100)
        batch_X, batch_Y, totals = [], [], 0
        bar = tqdm.tqdm(total=total_steps, desc="Training TunedLens(Layer=%d)" % layer)
        with open(corpus, encoding="utf8") as f:
            out_emb = self._out_emb.cuda()
            for text in f:
                text = render(text)
                with tc.no_grad():
                    hiddens, logits = self.get_activates(text, return_logits=True)
                    batch_X.append(hiddens[layer].cpu().squeeze())
                    batch_Y.append(logits.cpu().squeeze())
                totals += len(batch_Y[-1])

                if totals >= batch_size:
                    optim.zero_grad()
                    with tc.no_grad():
                        X = tc.vstack(batch_X).cuda()
                        Y = tc.vstack(batch_Y).cuda()
                    loss = self._translator.compute_loss(X, Y, out_emb, T)
                    history.append(loss.item())
                    loss.backward()
                    self._translator.clip_grad(1.0)
                    optim.step()
                    batch_X, batch_Y, totals = [], [], 0
                    bar.update(1)
                    if bar.n % 200 == 0:
                        print("Step=%d | Score=%.4f" % (bar.n, history.average()))
                    if bar.n == total_steps:
                        break
                    if history.count_trends(3, eps) == 3:
                        print("Early stopping at Step=%d with Score=%.4f" % (bar.n, history.average()))
                        break
        self._translator.dump_disk(fpath)

    @tc.no_grad()
    def generate(self, text, **kwrds):
        ids = tc.tensor([self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))]).to(self._device)
        tokens = self._model.generate(ids, pad_token_id=self._tokenizer.eos_token_id, **kwrds)[:, ids.shape[1]:]
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    def explain(self, embeds, topK=50, alpha=1.0, beta=1.0, **kwrds):
        #assert isinstance(embeds, tc.Tensor)
        #assert len(embeds.shape) == 2
        if self._translator:
            self._translator.eval()
            embeds = self._translator(embeds) 
        embeds = embeds.float().cpu()

        with tc.no_grad():
            # arg max MI(W; C) = arg max H(C) - H(C|W)
            #                  = arg min H(C|W)
            #                  = arg min - \sum_{w\in W}\sum_{c\in C} p(c, w) * log p(c|w)
            #                  = arg min - \sum_{w\in W}\sum_{c\in C} p(c) * p(w|c) * log p(c|w)
            #                  = arg min - \sum_{w\in W}\sum_{c\in C} p(w|c) * log p(c|w)
            #                  = arg max \sum_{w\in W}\sum_{c\in C} p(w|c) * log p(c|w)
            cooccurent = embeds @ self._vocab.T 
            term_freq = tc.softmax(cooccurent / alpha, 1)
            conc_freq = tc.log_softmax(cooccurent / beta, 0)
            scores = term_freq * conc_freq
        
        for idx, (term, conc, scores) in enumerate(zip(term_freq, conc_freq, scores)):
            # the following line does arg min
            pairs = sorted(zip(scores.tolist(), self._words, term.tolist(), conc.tolist()), reverse=True)
            words = []
            for s, w, p, c in pairs:
                words.append(w) #+ "(%.8f|%.8f,%.8f)" % (s, p, c))
                if len(words) == topK:
                    break
            yield words 

def compute_proba(x, dim=-1, eps=0.1):
    prior = eps / x.shape[dim]
    proba = tc.softmax(x, dim)
    return prior + (1.0 - eps) * proba


def topk_softmax(x, k: int, dim = -1) :
    """
    Compute softmax over the top-k values along a given dimension.
    Non-top-k entries are set to 0.

    Args:
        x (torch.Tensor): Input tensor.
        k (int): Number of top elements to consider.
        dim (int): Dimension along which to apply top-k softmax.

    Returns:
        torch.Tensor: Same shape as x, with softmax probabilities on top-k entries and 0 elsewhere.
    """
    # Get top-k values and indices
    topk_vals, topk_idx = tc.topk(x, k, dim=dim)

    # Apply softmax over top-k values
    topk_probs = tc.softmax(topk_vals, dim=dim)

    # Scatter back into full tensor shape
    out = tc.zeros_like(x).scatter(dim, topk_idx, topk_probs)
    return out

def render(text):
    text = text.strip()
    for key, val in [("\\n", "\n"), ("\\t", "\t")]:
        text = text.replace(key, val)
    return "<s>[INST] %s [/INST]" % text




if __name__ == "__main__":
    name, layer, sae = load_pretrained("./outputs/weights/TopK7_l8_h65k_epoch5.pth")
    print("Running on Hidden Layer %d with %s_SAE." % (layer, name))

    explainer = MIExplainer("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    explainer.build_translator(corpus="datasets/prompt_dataset.tsv", layer=layer, batch_size=32, learn_rate=1e-3, total_steps=3000, name="mistral")


    words = load_wiki_words("./prompt_vocab_naive.txt")[:5000] 
    explainer.build_vocab(words)
    
    W = sae.W_enc.T
    from collections import Counter
    with open("outputs/annotations/MI_%s_MI_wordlist.txt" % name, 'w', encoding='utf8') as f:
        bar = tqdm.tqdm(total=W.shape[0])
        uniques = Counter()
        f.write("FeatureID\tWords\n")
        for idx, exp in enumerate(explainer.explain(W)):
            #if idx > 100:
            #    break
            uniques.update((_.split("(", 1)[0] for _ in exp))
            line = [_.replace('\n', '\\n').replace('\t', '\\t') for _ in (str(idx), "|||".join(exp))]
            #print(idx, "-->",  ", ".join(exp))
            f.write('\t'.join(line) + '\n')
            bar.update(1)
    superficial = {_ for i, _ in enumerate(words) if i < 250 or len(_) < 4}
    diverse = sum(1 for _ in uniques if _ not in superficial) / (len(words) - len(superficial))
    lexical = sum(1 for _ in uniques if _ in superficial) / len(superficial)
    dominate = sum(1 for _ in uniques.values() if _ / idx > 0.1) / len(uniques)
    print(u"Diversity↑=%.4f" % diverse)
    print(u"Lexical↓=%.4f" % lexical)
    print(u"Dominate↓=%.4f" % dominate)
    print("Total=%.4f" % (diverse - lexical))
