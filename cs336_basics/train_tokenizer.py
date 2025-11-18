from collections import Counter
import os
import regex as re
from typing import BinaryIO
import multiprocessing as mp


NUM_BYTE_VALUES = 256
TINY_STORIES_VAL_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
PRE_TOKENIZE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKENIZER_RE = re.compile(PRE_TOKENIZE_PAT)

END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_BYTES = END_OF_TEXT.encode()


ByteTuple = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe_slow(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytePair]]:
    pretoken_counts = get_pretoken_counts(input_path, special_tokens=special_tokens)

    merges = []
    vocab = _init_vocab(special_tokens)
    while len(vocab) + len(merges) < vocab_size:
        pair_counts = count_byte_pairs(pretoken_counts)
        pair_to_merge = get_byte_pair_to_merge(pair_counts)
        pretoken_counts = merge_pretoken_counts(
            pair_to_merge=pair_to_merge,
            pretoken_counts=pretoken_counts,
        )
        merges.append(pair_to_merge)
    
    for i, (m1, m2) in enumerate(merges, len(vocab)):
        vocab[i] = m1 + m2

    return vocab, merges

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a (byte-level) BPE tokenizer.

    Args:
      input_path: Path to a text file with BPE tokenizer training data.
      vocab_size: A positive int that defines the maximum final vocabulary size
                  (including the initial byte vocabulary, vocabulary items
                  produced from merging, and any special tokens).
      special_tokens: A list of strings to add to the vocabulary.
                      These special tokens do not otherwise affect BPE training.

    Returns:
      vocab: The tokenizer vocabulary, a mapping from int
             (token id in the vocabulary) to bytes (token bytes).
      merges: A list of BPE merges produced from training. Each list item is a
              tuple of bytes (<token1>, <token2>), representing that <token1>
              was merged with <token2>.
              The merges are ordered by order of creation.
    """


def _init_byte_vocab() -> dict[int, bytes]:
    byte_vocab = {i: bytes([i]) for i in range(NUM_BYTE_VALUES)}
    return byte_vocab


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = _init_byte_vocab()
    for i, special_token in enumerate(special_tokens, len(vocab)):
        vocab[i] = special_token.encode()
    return vocab


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join(map(re.escape, special_tokens))
    return re.split(pattern, text)


def get_chunk_boundaries(filepath: str, desired_num_chunks: int, split_special_token: bytes = END_OF_TEXT_BYTES) -> list[int]:
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=desired_num_chunks, split_special_token=split_special_token)
        return boundaries


def read_chunk(file: BinaryIO, start: int, end: int, encoding: str = "utf-8", errors: str = "ignore") -> str:
    file.seek(start)
    chunk = file.read(end - start).decode(encoding=encoding, errors=errors)
    return chunk


def get_pretoken_counts(filepath: str, special_tokens: list[str], num_processes: int | None = None) -> Counter[tuple[bytes, ...]]:
    if num_processes is None:
        num_processes = mp.cpu_count()
    boundaries = get_chunk_boundaries(filepath, num_processes)
    tasks = [(start, end, special_tokens, filepath) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(num_processes) as p:
        counters = p.starmap(process_chunk_boundary, tasks)
        pretoken_counts = Counter()
        for counter in counters:
            pretoken_counts.update(counter)
        return pretoken_counts


def process_chunk_boundary(start: int, end: int, special_tokens: list[str], filepath: str, encoding: str = "utf-8") -> Counter[tuple[bytes, ...]]:
    counter = Counter()

    with open(filepath, "rb") as f:
        chunk = read_chunk(f, start, end)
        subchunks = _split_on_special_tokens(chunk, special_tokens=special_tokens)
        for subchunk in subchunks:
            for m in TOKENIZER_RE.finditer(subchunk):
                pretoken = m.group().encode(encoding)
                pretoken_tuple = tuple(bytes([b]) for b in pretoken)
                counter[pretoken_tuple] += 1

    return counter


def count_byte_pairs(pretoken_counts: Counter[ByteTuple]) -> Counter[BytePair]:
    pair_counts = Counter()

    for s, cnt in pretoken_counts.items():
        for i in range(len(s) - 1):
            pair = (s[i], s[i+1])
            pair_counts[pair] += cnt
    
    return pair_counts


def get_byte_pair_to_merge(byte_pair_counts: Counter[BytePair]) -> BytePair:
    pair_to_merge, _ = max(byte_pair_counts.items(), key=lambda bp: (bp[1], bp[0]))
    return pair_to_merge


def merge_pretoken_counts(pair_to_merge: BytePair, pretoken_counts: Counter[ByteTuple]) -> Counter[ByteTuple]:
    merged_pretoken_counts = Counter()
    for s, v in pretoken_counts.items():
        merged = []
        i = 0
        while i < len(s):
            if i < len(s) - 1 and s[i:i+2] == pair_to_merge:
                merged.append(s[i] + s[i+1])
                i += 2
            else:
                merged.append(s[i])
                i += 1
        merged = tuple(merged)
        merged_pretoken_counts[merged] = v
    
    return merged_pretoken_counts


if __name__ == "__main__":
    pretoken_counts = get_pretoken_counts(TINY_STORIES_VAL_PATH, special_tokens=[END_OF_TEXT])
    print(pretoken_counts)