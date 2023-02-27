import re
import os

import re
import os
import re
import math
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer = PorterStemmer()
stop_words = set((stopwords.words("english")) + [",", ":"])


def normalise_query(query):
    query = clean_line(query)
    tokens = word_tokenize(query.lower())
    filtered_tokens = [
        stemmer.stem(token) for token in tokens if token not in stop_words
    ]
    return " ".join(filtered_tokens)


def clean_line(line):
    line = re.sub(r"[_.]{2,}", "", line)
    line = re.sub(r"[\"]", "", line)
    line = re.sub(r"\w *\)", "", line)
    line = re.sub(r"^\d *\.", "", line)
    return line


# def stemmed(paragraph):


def split_into_passages(filename, path):
    processed_filename = filename.replace(".txt", "_processed.txt")
    output_file = open("Unnormal_new/" + processed_filename, "w")
    paragraph = ""
    current_word_count = 0
    with open(path, "r") as f:
        for line in f:
            cleaned = clean_line(line)
            stripped_line = cleaned.strip()
            if len(stripped_line) > 2:
                paragraph += cleaned
                current_word_count += len(cleaned.split())
            elif re.search("[ \r\t\n]*", line):
                if current_word_count > 20:
                    output_file.write(normalise_query(paragraph) + "$$$\n")
                    paragraph = ""
                    current_word_count = 0
    output_file.close()


def unnormal_to_processed(filename, path):
    processed_filename = filename  # .replace("_processed.txt", "_normal.txt")
    output_file = open("Normal/" + processed_filename, "w")
    paragraph = ""
    with open(path, "r") as f:
        passages = f.read().split("$$$")
        for passage in passages:
            output_file.write(normalise_query(passage) + "$$$\n")
    output_file.close()


for filepath in os.listdir("Unnormal/"):
    unnormal_to_processed(filepath, "Unnormal/" + filepath)
    # split_into_passages(filepath, "Originals/" + filepath)