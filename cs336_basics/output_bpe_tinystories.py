from cs336_basics.train_tokenizer import load_obj


def get_longest_token(vocab: dict[int, bytes]) -> bytes:
    return max(vocab.values(), key=lambda x: len(x))


if __name__ == "__main__":
    vocab = load_obj("vocab.pkl")
    merges = load_obj("merges.pkl")
    longest_token = get_longest_token(vocab)
    print(longest_token)
