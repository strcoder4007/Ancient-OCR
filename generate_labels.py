def duplicate_words(input_file, output_file, repeat_count=10):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            words = infile.readlines()
            
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, word in enumerate(words):
                word = word.strip()
                for count in range(repeat_count):
                    outfile.write(f"word_{i}_aug_{count}.png {word}\n")

        print(f"Successfully wrote {len(words) * repeat_count} words to {output_file}.")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "kaithi_1000.txt"
    output_file = "labels_1000.txt"
    duplicate_words(input_file, output_file)