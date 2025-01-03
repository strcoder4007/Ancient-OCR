import os
from datasets import load_dataset
from collections import Counter
import re

# Load the dataset
dataset = load_dataset('zicsx/Wikipedia-Hindi')

def tokenize(text):
    return text.lower().split()

word_counts = Counter()

for split in dataset.keys():
    for entry in dataset[split]:
        words = tokenize(entry['text'])
        word_counts.update(words)

top_words = [word for word, _ in word_counts.most_common(50000)]


output_file = 'top_50000_words.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for word in top_words:
        f.write(word + '\n')

print(f"The top 50,000 words have been written to {output_file}.")
