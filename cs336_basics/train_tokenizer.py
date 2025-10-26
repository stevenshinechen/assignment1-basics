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
  pass
  