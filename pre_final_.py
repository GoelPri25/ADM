
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import time



print("Load data")
data = np.load("./user_movie_rating.npy")


np.random.seed(5)


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
num_hashmap = 100
# Minhash and Similarity Matrix
hashes = np.array([np.random.permutation(num_movies) for _ in range(num_hashmap)])
signature_matrix = np.zeros((num_hashmap, num_users))

non_zero_indices = [row.nonzero()[1] for row in ratings_matrix]

def get_signature_minhashing(num_hashmap, non_zero_indicies, num_users, hashes, signature_matrix):
    for i in range(num_hashmap):  # maybe try mulit process
        for j in range(num_users):
            non_zero_columns = non_zero_indices[j]
            #none_zero_columns = ratings_matrix[j].nonzero()[1]
            if len(non_zero_columns) > 0:
                min_hash_index = np.argmin(hashes[i, non_zero_columns])
                
                # signature_matrix[i, j] = min_hash_index
                signature_matrix[i, j] = hashes[i, non_zero_columns[min_hash_index]]
            else:
                signature_matrix[i, j] = float("inf")
            
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_hashmap} hash functions.")

    print(f"MinHash signatures computed. Spent {time.time() - start_time} seconds.")
    return signature_matrix
# Sort signature values for each user (j) across all hash functions (i) # will result in more pairs 
# for j in range(num_users):
#     signature_matrix[:, j] = np.sort(signature_matrix[:, j])  # Sort column for user j


#beta = int(r/2)
def lsh_candidate_pairing(b, r, signature_matrix):
    candidate_pairs = set()
    for i in range(b):
        # bands.append(signature_matrix[i * 5 :(i + 1) * 5])
        if i == b-1:
            band = signature_matrix[i * r :]
        else:
            band = signature_matrix[i * r : (i + 1) * r]
        
        band_buckets = defaultdict(list)
        for user in range(num_users):
            
            #band_hash = hash(tuple(sorted(set(band[:, user]))[:beta]))
            band_hash = hash(tuple(sorted(set(band[:, user]))))
            
            band_buckets[band_hash].append(user)
            
        for bucket_users in band_buckets.values():
            # if len(bucket_users) > 1 and len(bucket_users) < 0.5 * num_users:
            if len(bucket_users) > 1:
                for i in range(len(bucket_users)):
                    for j in range(i + 1, len(bucket_users)):
                        
                        pair = (min(bucket_users[i], bucket_users[j]), max(bucket_users[i], bucket_users[j]))  #  consistent ordering
                        if pair not in candidate_pairs:  # Check if the pair already exists
                            candidate_pairs.add(pair)
                        #candidate_pairs.add((bucket_users[i], bucket_users[j]))
    return candidate_pairs

def compute_similarity(pair, non_zero_indices):
    user1, user2 = pair
    movies1 = set(non_zero_indices[user1])
    movies2 = set(non_zero_indices[user2])
    intersection = len(movies1 & movies2)
    union = len(movies1 | movies2)
    return (user1, user2, intersection / union if union > 0 else 0)


def get_jaccards_pairs(candidate_pairs):
    jaccard_similarities_pairs = []
    for user1, user2 in candidate_pairs:
        jaccard_sim = compute_similarity((user1, user2), non_zero_indices)
        jaccard_similarities_pairs.append(jaccard_sim)
    return jaccard_similarities_pairs

def get_filtered_pairs(jaccard_similarities_pairs, threshold=0.5):
    filtered_pairs = []
    for i in range(len(jaccard_similarities_pairs)):
        if jaccard_similarities_pairs[i][2] > threshold:
            filtered_pairs.append(jaccard_similarities_pairs[i])
    return filtered_pairs




# Plot pictures for the report. Should remove before submit.
import matplotlib.pyplot as plt



# Write Results
def write_results(pairs, output_file):
    """Write results to a file."""
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        for pair in sorted(pairs):
            f.write(f"{pair[0]},{pair[1]}\n")
    print("Results written successfully.")
output_file = 'result.txt'

print("Using locality sensitive hashing")
signature_matrix = get_signature_minhashing(num_hashmap=num_hashmap, non_zero_indicies=non_zero_indices, num_users= num_users, hashes= hashes, signature_matrix= signature_matrix)



start_time = time.time()

r = 8  # when you work with signatures of length n = 100, you may consider e.g., b = 3, r = 33; b = 6, r = 15, etc.
b = num_hashmap // r




# bands = []

candidate_pairs = lsh_candidate_pairing(b,r,signature_matrix)



print(f"Candidate pairs selected. Spent {time.time() - start_time} seconds.")

print(f"Number of candidate pairs: {len(candidate_pairs)}")

print("Compute similarity on candidate pairs")
start_time = time.time()

print(f"Finished similarity computation. Spent {time.time() - start_time} seconds")

threshold = 0.5
print(f"Filter similarity < {threshold}")
start_time = time.time()
final_pairs = []


jaccad_scores_pairs = get_jaccards_pairs(candidate_pairs)

final_pairs = get_filtered_pairs(jaccad_scores_pairs)

final_pairs.sort(key=lambda x: x[2])

similarity_scores = [pair[2] for pair in final_pairs]
write_results(final_pairs, output_file)

plt.figure(figsize=(8, 6))
plt.scatter(
    range(1, len(similarity_scores) + 1),
    similarity_scores,
    color="blue",
    s=10,
    label="Jaccard Similarity",
)

plt.title(f"{len(similarity_scores)} pairs with JS > 0.5", fontsize=14)
plt.xlabel("Most similar pairs", fontsize=12)
plt.ylabel("Jaccard Similarity", fontsize=12)
plt.ylim(0.5, 1.0)
plt.xlim(0, len(final_pairs) * 1.1)
plt.legend("")
plt.grid(True)
plt.show()


print(f"Spent {time.time() - start_time} seconds on similarity filtering.")
print(f"Number of pairs > threshold {threshold} = {len(final_pairs)}")
