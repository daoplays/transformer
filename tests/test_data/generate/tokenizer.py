import json
import regex as re
from typing import List, Dict, Tuple
from transformers import GPT2Tokenizer

class GPT2TokenizerFromScratch:
    def __init__(self, vocab_file: str, merges_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Add byte-level encoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def bpe(self, token: str) -> str:
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            for pair in pairs:
                print("pairs ", pair, self.bpe_ranks.get(pair, float('inf')))
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        return ' '.join(word)

    @staticmethod
    def get_pairs(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def tokenize(self, text: str) -> List[str]:
        bpe_tokens = []
        print("pat tokens: ")
        print(re.findall(self.pat, text))
        for token in re.findall(self.pat, text):
            print(token.encode('utf-8'))
            for b in token.encode('utf-8'):
                print(b, self.byte_encoder[b])
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            print("token", token)
            print("bpe token", self.bpe(token))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# Usage
root = "/home/ltl/.cache/huggingface/transformers/gpt2/"
input_string = "GPT2 is a model developed by OpenAI"

tokenizer = GPT2TokenizerFromScratch(root+'vocab.json', root+'merges.txt')
tokens = tokenizer.tokenize(input_string)

# Compare with Hugging Face implementation
hf_tokenizer = GPT2Tokenizer.from_pretrained(root)
hf_tokens = hf_tokenizer.tokenize(input_string)
hf_token_ids = hf_tokenizer.convert_tokens_to_ids(hf_tokens)

# Compare results
print(f"Our implementation: {tokens}")
print(f"Hugging Face implementation: {hf_tokens}")
print(f"Token IDs: {hf_token_ids}")
print(f"Tokens match: {tokens == hf_tokens}")