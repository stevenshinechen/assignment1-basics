from collections import Counter

EXAMPLE_CORPUS = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

def pretokenize_whitespace(text: str):
    return text.split()

def count_byte_pairs(word_counts: Counter) -> Counter:
    byte_pair_counts = Counter()

    for s, cnt in word_counts.items():
        for i in range(len(s) - 1):
            byte_pair_counts[s[i:i+2]] += cnt
    
    return byte_pair_counts

def get_byte_pair_to_merge(byte_pair_counts: Counter) -> tuple[bytes, int]:
    bp_to_merge, bp_to_merge_cnt = max(byte_pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
    return bp_to_merge, bp_to_merge_cnt

def merge_pretoken_counts(bp_to_merge: tuple[bytes, bytes], pretoken_counts: Counter) -> Counter:
    merged_pretoken_counts = Counter()
    for s, v in pretoken_counts.items():
        merged = []
        i = 0
        while i < len(s):
            if i < len(s) - 1 and s[i:i+2] == bp_to_merge:
                merged.append("".join(s[i:i+2]))
                i += 2
            else:
                merged.append(s[i])
                i += 1
        merged = tuple(merged)
        merged_pretoken_counts[merged] = v
    
    return merged_pretoken_counts

def make_vocab(merges):
    vocab = [b"<|endoftext|>"] + [bytes([i]) for i in range(256)]
    for merge in merges:
        vocab.append("".join(merge).encode())

    return vocab

if __name__ == "__main__":
    pretokens = pretokenize_whitespace(EXAMPLE_CORPUS)
    word_pretoken_counts = Counter(pretokens)

    pretoken_counts = Counter({tuple(k): v for k, v in word_pretoken_counts.items()})

    print(f"{word_pretoken_counts=}")

    merges = []
    for i in range(6):
        byte_pair_counts = count_byte_pairs(pretoken_counts)

        bp_to_merge, bp_to_merge_cnt = get_byte_pair_to_merge(byte_pair_counts)

        merged_pretoken_counts = merge_pretoken_counts(bp_to_merge, pretoken_counts)

        print(f"==== Iteration {i+1} ====")
        print(f"{pretoken_counts=}")
        print(f"{byte_pair_counts=}")
        print(f"{bp_to_merge=}, {bp_to_merge_cnt=}")
        print(f"{merged_pretoken_counts=}")
        print()
        pretoken_counts = merged_pretoken_counts
        merges.append(bp_to_merge)

    print(f"{merges=}")
    vocab = make_vocab(merges)
    print(f"{vocab=}")