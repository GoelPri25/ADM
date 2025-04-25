import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import time

print("Load data")
data = np.load("./user_movie_rating.npy")

user_ids = data[:, 0]
movie_ids = data[:, 1]
ratings = data[:, 2]
# Mapping ID to index
if len(user_ids) - max(user_ids) == 1:
    row_indices = np.array([i for i in range(len(user_ids))])
    col_indices = np.array([i for i in range(len(movie_ids))])

else:
    user_map = {uid: idx for idx, uid in enumerate(np.unique(user_ids))}
    movie_map = {mid: idx for idx, mid in enumerate(np.unique(movie_ids))}

    row_indices = np.array([user_map[uid] for uid in user_ids])
    col_indices = np.array([movie_map[mid] for mid in movie_ids])

print("Transfer to sparse matrix")
ratings_matrix = csr_matrix((ratings, (row_indices, col_indices)))

# print("Sparse Matrix Shape:", ratings_matrix.shape)
# print("Non-zero Entries:", ratings_matrix.nnz)
# print("Matrix in Dense Format:\n", ratings_matrix.toarray())
print("Using MinHash to create signature matrix")
start_time = time.time()
num_users = len(np.unique(user_ids))
num_movies = len(np.unique(movie_ids))
num_hashmap = 90

# Minhash and Similarity Matrix
hashes = np.array([np.random.permutation(num_movies) for _ in range(num_hashmap)])
signature_matrix = np.zeros((num_hashmap, num_users))

# for i in range(num_hashmap):
#     none_zero_columns = ratings_matrix.nonzero()[1]
#     signature_matrix[i, :] = np.min(hashes[i, none_zero_columns])
#     if (i + 1) % 10 == 0:
#         print(f"Completed {i + 1}/{num_hashmap} hash functions.")

for i in range(num_hashmap):
    for user in range(num_users):
        none_zero_column = ratings_matrix[user].nonzero()[1]
        if len(none_zero_column) > 0:
            for movie in hashes[i]:
                if movie in none_zero_column:
                    signature_matrix[i, user] = movie
                    break
        else:
            signature_matrix[i, user] = -1
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{num_hashmap} hash functions.")

print(f"MinHash signatures computed. Spent {time.time() - start_time} seconds.")

print("Using locality sensitive hashing")
start_time = time.time()
r = 15  # when you work with signatures of length n = 100, you may consider e.g., b = 3, r = 33; b = 6, r = 15, etc.
b = num_hashmap // r

# bands = []
candidate_pairs = set()
for i in range(b):
    # bands.append(signature_matrix[i * 5 :(i + 1) * 5])
    band = signature_matrix[i * 5 : (i + 1) * 5]
    band_buckets = defaultdict(list)
    for user in range(num_users):
        band_hash = hash(tuple(band[:, user]))
        band_buckets[band_hash].append(user)

    for bucket_users in band_buckets.values():
        if len(bucket_users) > 1:
            for i in range(len(bucket_users)):
                for j in range(i + 1, len(bucket_users)):
                    candidate_pairs.add((bucket_users[i], bucket_users[j]))

print(f"Candidate pairs selected. Spent {time.time() - start_time} seconds.")

print(f"Number of candidate pairs: {len(candidate_pairs)}")

print("Compute similarity on candidate pairs")
start_time = time.time()


def compute_similarity(pair, sparse_matrix):
    user1, user2 = pair
    movies1 = set(sparse_matrix[user1].nonzero()[1])
    movies2 = set(sparse_matrix[user2].nonzero()[1])
    intersection = len(movies1 & movies2)
    union = len(movies1 | movies2)
    return (user1, user2, intersection / union if union > 0 else 0)


jaccard_similarities = []
for user1, user2 in candidate_pairs:
    jaccard_sim = compute_similarity((user1, user2), ratings_matrix)
    jaccard_similarities.append((user1, user2, jaccard_sim))

print(f"Finished similarity computation. Spent {time.time() - start_time} seconds")

threshold = 0.5
print(f"Filter similarity < {threshold}")
start_time = time.time()
filted_pairs = []
for i in range(len(jaccard_similarities)):
    if jaccard_similarities[i][2] >= threshold:
        filted_pairs.append(jaccard_similarities[i])

print(f"Spent {time.time() - start_time} seconds on similarity filtering.")
print(f"Number of pairs > threshold {threshold} = {len(filted_pairs)}")

# Plot pictures for the report. Should remove before submit.
import matplotlib.pyplot as plt

filted_pairs.sort(key=lambda x: x[2])

similarity_scores = [pair[2] for pair in filted_pairs]

plt.figure(figsize=(8, 6))
plt.scatter(
    range(1, len(similarity_scores) + 1),
    similarity_scores,
    color="blue",
    s=10,
    label="Jaccard Similarity",
)

plt.title(f"{len(similarity_scores)} pairs with JS >= 0.5", fontsize=14)
plt.xlabel("Most similar pairs", fontsize=12)
plt.ylabel("Jaccard Similarity", fontsize=12)
plt.ylim(0.5, 1.0)
plt.xlim(0, len(filted_pairs) * 1.1)
plt.grid(True)
plt.show()
