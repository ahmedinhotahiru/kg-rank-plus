import os
import pandas as pd

# Import the necessary functions from moverscore_v2 (the fast version)
from moverscore import get_idf_dict, word_mover_score

# Path to your CSV file; update as needed.
csv_file = "/ocean/projects/cis240101p/aissah/KG-Rank-main/results/LiveQA_gpt4_random_results_20250413_144830.csv"

# Read the CSV file using pandas. The CSV should include columns: 
# "Generated_Answer" and "Reference_Answer"
df = pd.read_csv(csv_file)

# Convert the "Generated_Answer" and "Reference_Answer" columns to Python lists.
generated_answers = df["Generated_Answer"].astype(str).tolist()
reference_answers = df["Reference_Answer"].astype(str).tolist()

# Build the IDF dictionaries from both the reference and generated answers.
idf_dict_ref = get_idf_dict(reference_answers)
idf_dict_gen = get_idf_dict(generated_answers)

# Calculate the MoverScore.
# Options:
# - stop_words: list of words to ignore (set to empty list here)
# - n_gram: set to 1 for unigram-based MoverScore (can change to 2 for bigrams)
# - remove_subwords: True to clean up subword artifacts.
scores = word_mover_score(
    reference_answers,
    generated_answers,
    idf_dict_ref,
    idf_dict_gen,
    stop_words=[],     # specify stopwords if necessary
    n_gram=1,          # using unigrams
    remove_subwords=True
)

# Optionally compute the average MoverScore
average_score = sum(scores) / len(scores) if scores else 0

print("MoverScore per example:")
print(scores)
print("\nAverage MoverScore over the dataset:")
print(average_score)
