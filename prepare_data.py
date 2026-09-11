import pandas as pd
import numpy as np

# Step A: build the set of original EMNIST labels we want to keep
# Digits 1-9 are labels 1-9, uppercase A-Z are labels 10-35
target_labels = list(range(1, 10)) + list(range(10, 36))

# Step B: build old_label -> new_label (0-34) mapping
# sorted() ensures a consistent, predictable order
old_to_new = {old_label: new_index for new_index, old_label in enumerate(sorted(target_labels))}
print("Label mapping (old EMNIST label -> new class index):")
print(old_to_new)

# Step C: read the file in chunks, collect samples until each class has enough
SAMPLES_PER_CLASS = 100  # buffer above the required 50, gives room for train/val/test split
chunk_size = 10000

collected_rows = []
counts = {label: 0 for label in target_labels}

for chunk in pd.read_csv('data/emnist-byclass-train.csv', header=None, chunksize=chunk_size, dtype=np.uint8):
    for label in target_labels:
        if counts[label] >= SAMPLES_PER_CLASS:
            continue  # already have enough of this class, skip
        # find rows in this chunk matching this label, that we still need
        needed = SAMPLES_PER_CLASS - counts[label]
        matches = chunk[chunk[0] == label].head(needed)
        if len(matches) > 0:
            collected_rows.append(matches)
            counts[label] += len(matches)

    # stop early once every class is full
    if all(c >= SAMPLES_PER_CLASS for c in counts.values()):
        print("Collected enough samples for all classes, stopping early.")
        break

print("Final counts per class:", counts)

# Combine all the small chunks into one big DataFrame
all_data = pd.concat(collected_rows, ignore_index=True)

# Separate labels (column 0) from pixels (columns 1-784)
y_old_labels = all_data[0].values           # shape (3500,)
X = all_data.drop(columns=[0]).values       # shape (3500, 784)

# Apply our old_label -> new_label (0-34) mapping
y = np.array([old_to_new[label] for label in y_old_labels])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("y unique values:", sorted(set(y)))
print("y min/max:", y.min(), y.max())

# Normalize pixel values to 0-1 range
X = X.astype(np.float32) / 255.0

# Shuffle X and y together (same random order for both, so labels stay matched to images)
np.random.seed(42)  # fixes the randomness so results are reproducible - important for your report
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Split: 70% train, 15% validation, 15% test
n = len(X)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Save everything so we never need to re-read the CSVs again
np.savez('data/processed_data.npz',
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)
print("Saved to data/processed_data.npz")
