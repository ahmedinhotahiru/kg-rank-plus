import pandas as pd
from moverscore_v2 import word_mover_score, get_idf_dict
import numpy as np

# Load and clean data
df = pd.read_csv("/ocean/projects/cis240101p/aissah/KG-Rank-main/results/LiveQA_gpt4_random_results_20250413_144830.csv").dropna(subset=["Generated_Answer", "Reference_Answer"])

# Convert all answers to strings and handle empty cases
df = df.map(lambda x: str(x) if pd.notnull(x) else "")
df = df[(df["Generated_Answer"].str.len() > 0) & (df["Reference_Answer"].str.len() > 0)]

# Compute MoverScore with modern BERT implementation
try:
    idf_dict_ref = get_idf_dict(df["Reference_Answer"].tolist())  # From reference texts
    idf_dict_hyp = get_idf_dict(df["Generated_Answer"].tolist())  # From generated texts

    scores = word_mover_score(
        df["Reference_Answer"].tolist(),
        df["Generated_Answer"].tolist(),
        idf_dict_ref,
        idf_dict_hyp,
         # stop_words=[],  # Uncomment if you have stop words
         #  1,  # n_gram
         #  True  # remove_subwords
         #  device="cuda",  # Uncomment if you want to use GPU
         #  8,  # Reduce for memory constraints
         #  1,  # n_gram
         #  True  # remove_subwords
         #  device="cuda",  # Uncomment if you want to use GPU
        batch_size=8,  # Reduce for memory constraints
        n_gram=1,
        remove_subwords=True
    )
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("Reduce batch_size or use CPU")
        scores = word_mover_score(
            df["Reference_Answer"].tolist(),
            df["Generated_Answer"].tolist(),
            batch_size=1,
            device="cpu",
            n_gram=1
        )
    else:
        raise

df["MoverScore"] = np.round(scores, 4)
df.to_csv("../evaluation_results/scored_results_fixed.csv", index=False)
