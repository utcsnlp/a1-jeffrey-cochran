# constants.py

from numpy import zeros

permissible_labels = [
    "O",
    "ORG",
    "MISC",
    "PER",
    "LOC"
]

num_permissible_labels = len(permissible_labels)

label_idx = { k: v for (v, k) in enumerate(permissible_labels)}
id_labels = { k: v for (k, v) in enumerate(permissible_labels)}

label_vectors = {}
for k, v in label_idx.items():
    current_vector = zeros((num_permissible_labels,))
    current_vector[v] = 1.
    label_vectors[k] = current_vector