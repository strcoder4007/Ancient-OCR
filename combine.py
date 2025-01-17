import random

# Function to read strings from a file
def read_strings_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# Function to randomly combine the strings
def random_combinations(strings, min_size=2, max_size=3):
    random.shuffle(strings)  # Shuffle the strings to ensure randomness
    combinations = []

    i = 0
    while i < len(strings):
        size = random.randint(min_size, max_size)  # Decide whether to take a pair or triplet
        if i + size > len(strings):  # Ensure we don't go out of bounds
            size = len(strings) - i
        combinations.append(" ".join(strings[i:i + size]))
        i += size

    return combinations

# Function to write the output to a file or print to console
def write_combinations_to_file(combinations, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as file:
        for combination in combinations:
            file.write(combination + "\n")

# Main function to tie everything together
def main(input_filename, output_filename):
    strings = read_strings_from_file(input_filename)
    combinations = random_combinations(strings)
    write_combinations_to_file(combinations, output_filename)
    print("Combinations have been written to", output_filename)

input_filename = 'kaithi_5k.txt'
output_filename = 'combined_words.txt' 
main(input_filename, output_filename)
