import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    layout_file="layouts/notebook.slides.json",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    nomura_wiki="""Nomura was founded by Tokushichi Nomura, father of Nomura Securities founder Tokushichi Nomura II as a money changing business. This was just before the Meiji Restoration, the move to setting up a bank was a logical extension and progression of this business as times changed. Changes included the founding of stock exchanges in Tokyo and Osaka as the country became industrialised. Key amongst these changes was the Japanese government's decision to issue foreign currency denominated public bonds to fund the Russo-Japanese War; Nomura employed English speaking staff so that they could take on this international business.

    By 1906 Nomura had founded an in-house research department headed up by former Osaka newspaper journalist Kisaku Hashimoto. This was responsible for publishing the Osaka Nomura Business News with trading news, stock analysis and current economic trends. Research combined with a substantial newspaper advertising campaign helped raise the profile of Nomura. By 1917, Nomura had gone public and soon after Osaka Nomura Bank (the present day Resona Bank) was set up, within this business there was a securities section to handle bond sales and underwriting."""
    return (nomura_wiki,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Here are all the unique characters that occur in this text
    """)
    return


@app.cell
def _(nomura_wiki):
    chars = sorted(list(set(nomura_wiki)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)
    return (chars,)


@app.cell
def _(chars):
    # Naiive vocab based indexing
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    # encoder: take a string, output a list of integers
    encode = lambda s: [stoi[c] for c in s]
    # decoder: take a list of integers, output a string
    decode = lambda l: ''.join([itos[i] for i in l])

    print(encode("hii there"))
    print(decode(encode("hii there")))

    # [46, 47, 47, 1, 58, 46, 43, 56, 43]
    # hii there
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Byte pair encoding
    """)
    return


@app.cell
def _():
    def get_stats(ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts


    def merge(ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    return get_stats, merge


@app.cell
def _(get_stats, merge):
    class BasicTokenizer():

        def __init__(self):
            self.vocab = {}

        def train(self, text, vocab_size, verbose=False):
            assert vocab_size >= 256
            num_merges = vocab_size - 256

            # input text preprocessing
            text_bytes = text.encode("utf-8") # raw bytes
            ids = list(text_bytes) # list of integers in range 0..255

            # iteratively merge the most common pairs to create new tokens
            merges = {} # (int, int) -> int
            vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
            for i in range(num_merges):
                # count up the number of times every consecutive pair appears
                stats = get_stats(ids)
                # find the pair with the highest count
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = 256 + i
                # replace all occurrences of pair in ids with idx
                ids = merge(ids, pair, idx)
                # save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                # prints
                if verbose:
                    print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

            # save class variables
            self.merges = merges # used in encode()
            self.vocab = vocab   # used in decode()

        def decode(self, ids):
            # given ids (list of integers), return Python string
            text_bytes = b"".join(self.vocab[idx] for idx in ids)
            text = text_bytes.decode("utf-8", errors="replace")
            return text

        def encode(self, text):
            # given a string text, return the token ids
            text_bytes = text.encode("utf-8") # raw bytes
            ids = list(text_bytes) # list of integers in range 0..255
            while len(ids) >= 2:
                # find the pair with the lowest merge index
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                # subtle: if there are no more merges available, the key will
                # result in an inf for every single pair, and the min will be
                # just the first pair in the list, arbitrarily
                # we can detect this terminating case by a membership check
                if pair not in self.merges:
                    break # nothing else can be merged anymore
                # otherwise let's merge the best pair (lowest merge index)
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)
            return ids

    return (BasicTokenizer,)


@app.cell
def _(BasicTokenizer):
    # The Wikipedia example
    tokenizer = BasicTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
    print(tokenizer.encode(text))
    # [258, 100, 258, 97, 99]
    print(tokenizer.decode([258, 100, 258, 97, 99]))
    return (tokenizer,)


@app.cell
def _(nomura_wiki, tokenizer):
    tokenizer.train(nomura_wiki, 256 + 20, verbose=True) # 256 are the byte tokens, then do 3 merges
    # print(tokenizer.encode(nomura_wiki))
    return


@app.cell
def _(nomura_wiki, tokenizer):
    _encoded_nomura_wiki = tokenizer.encode(nomura_wiki)
    print(f"Length of text in characters: {len(nomura_wiki)}")
    print(f"Length of encoded text after merges: {len(_encoded_nomura_wiki)}")
    print(f"Efficiency gain: {len(nomura_wiki)/len(_encoded_nomura_wiki)}")
    return


@app.cell
def _(tokenizer):
    [tokenizer.decode([i]) for i in range(256,256+19)]
    return


if __name__ == "__main__":
    app.run()
