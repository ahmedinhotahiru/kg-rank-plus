# Optionally, set an alternative transformer model if desired.
# By default, the package uses an English BERT variant, which is limited to a maximum sequence length (typically 512 tokens).
import os
os.environ['MOVERSCORE_MODEL'] = "albert-base-v2"  # Uncomment or change if needed

# Import required functions from the fast moverscore implementation.
from moverscore import get_idf_dict, word_mover_score

# Sample reference texts (the "gold" outputs) and candidate texts (e.g., system-generated translations)
references = [
    "The cat sat on the mat.",
    "The cat is sitting on the mat."
]
translations = [
    "The cat sat on the mat.",
    "A cat is sitting on a rug."
]

# Create IDF dictionaries for both the reference corpus and the candidate (hypothesis) corpus.
# These dictionaries assign weights to words based on their frequency in the respective corpora.
idf_dict_ref = get_idf_dict(references)
idf_dict_hyp = get_idf_dict(translations)

# Compute the MoverScore.
# - stop_words: list of stop words to ignore (set to empty here).
# - n_gram: set to 1 for a unigram-based evaluation. You can set it to 2 for bigram-based evaluation.
# - remove_subwords: whether to remove subword artifacts like suffixes ("ING", "ED", etc.).
scores = word_mover_score(
    references, translations,
    idf_dict_ref, idf_dict_hyp,
    stop_words=[],     # specify any stopwords if desired
    n_gram=1,          # using unigrams for MoverScore
    remove_subwords=True
)

# Print the computed MoverScore(s)
print("MoverScore:", scores)
