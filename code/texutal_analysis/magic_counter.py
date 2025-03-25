import csv


def count_magic_words(csv_path):
    magic_words = [
        "and"
    ]
    '''
    magic_words = [
        "magic", "magical", "magician", "enchantment", "enchanted", "sorcery", "arcane",
        "mystical", "supernatural", "alchemy", "summoning", "prophecy", "omens", "incantation", "invocation",
        "evocation", "conjuring", "divination", "illusion", "wizard",
        "sorcerer", "mystic", "prophet"
    ]
    '''
    word_counts = {word: 0 for word in magic_words}
    rows_with_magic = 0

    with open(csv_path, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile, delimiter=';')
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue  # Skip malformed rows

            text = row[1].lower()
            found = False

            for word in magic_words:
                count = text.count(word)
                word_counts[word] += count
                if count > 0:
                    found = True

            if found:
                rows_with_magic += 1

    print(f"Total occurrences of magical words: {sum(word_counts.values())}")
    print(f"Number of images containing at least one magical word: {rows_with_magic}")
    for word, count in word_counts.items():
        if count > 0:
            print(f"{word}: {count}")


if __name__ == "__main__":
    input_csv = r"C:\Users\juliu\Documents\Coding Projects\Center for Digital Participation\toolify.ai_image_scraping\ai_website_images\cleaned_image_to_text_results.csv"
    count_magic_words(input_csv)
