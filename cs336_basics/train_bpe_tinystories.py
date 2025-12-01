import os
import time

import psutil
from cs336_basics.train_tokenizer import save_merges, save_vocab, train_bpe

def run_train_bpe(input_path: str = "data/TinyStoriesV2-GPT4-train.txt"):
    return train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

if __name__ == "__main__":
    process = psutil.Process(os.getpid())

    start_mem = process.memory_info().rss
    start_time = time.time()

    vocab, merges = run_train_bpe()

    end_time = time.time()
    end_mem = process.memory_info().rss

    save_vocab(vocab)
    save_merges(merges)

    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Memory before: {start_mem / (1024*1024):.2f} MiB")
    print(f"Memory after:  {end_mem / (1024*1024):.2f} MiB")

