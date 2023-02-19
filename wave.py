"""WaveNet"""
from data import def_lookup_table, get_dataset


if __name__ == "__main__":
    global lt, words

    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        lt = def_lookup_table(words)

        # Construct the dataset.
        x, labels = get_dataset(words, lt)
 
