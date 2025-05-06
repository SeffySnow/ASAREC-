import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def profiles(review, item_col, user_col, aspects):
    item_profile = review.groupby(item_col)[aspects].mean().reset_index()
    user_profile = review.groupby(user_col)[aspects].mean().reset_index()
    return user_profile, item_profile


import pandas as pd

# def profiles(review, item_col, user_col, aspects, threshold=1.5):
#     # --- Compute Raw Profiles ---
#     # Raw user profile: group by user and compute mean of aspects.
#     raw_user_profile = review.groupby(user_col)[aspects].mean().reset_index()
#     # Raw item profile: group by item and compute mean of aspects.
#     raw_item_profile = review.groupby(item_col)[aspects].mean().reset_index()
    
#     # --- Helper Function for Filtering ---
#     # Given a group (rows for an entity) and its raw mean, filter out rows that are
#     # very different (for every aspect, within raw_mean ± threshold * raw_std).
#     def filtered_mean_for_entity(group, raw_mean):
#         count = len(group)
#         if count > 5:
#             raw_std = group[aspects].std()
#             # Create a boolean mask: for each review, check that every aspect
#             # value is within [raw_mean - threshold*raw_std, raw_mean + threshold*raw_std].
#             mask = group[aspects].apply(lambda row: all(
#                 (row[a] >= raw_mean[a] - threshold * raw_std[a]) and 
#                 (row[a] <= raw_mean[a] + threshold * raw_std[a])
#                 if pd.notna(raw_std[a]) and raw_std[a] != 0 else True
#                 for a in aspects
#             ), axis=1)
#             filtered = group[mask]
#             if len(filtered) > 0:
#                 return filtered[aspects].mean()
#             else:
#                 return raw_mean
#         else:
#             return group[aspects].mean()
    
#     # --- Recompute User Profiles with Filtering ---
#     filtered_user_profiles = []
#     for user in raw_user_profile[user_col]:
#         group = review[review[user_col] == user]
#         # Retrieve the raw mean for this user from the raw user profile.
#         raw_mean = raw_user_profile.loc[raw_user_profile[user_col] == user, aspects].iloc[0]
#         mean_values = filtered_mean_for_entity(group, raw_mean)
#         profile = {user_col: user}
#         for a in aspects:
#             profile[a] = mean_values[a]
#         filtered_user_profiles.append(profile)
#     user_profile = pd.DataFrame(filtered_user_profiles)
    
#     # --- Recompute Item Profiles with Filtering ---
#     filtered_item_profiles = []
#     for item in raw_item_profile[item_col]:
#         group = review[review[item_col] == item]
#         raw_mean = raw_item_profile.loc[raw_item_profile[item_col] == item, aspects].iloc[0]
#         mean_values = filtered_mean_for_entity(group, raw_mean)
#         profile = {item_col: item}
#         for a in aspects:
#             profile[a] = mean_values[a]
#         filtered_item_profiles.append(profile)
#     item_profile = pd.DataFrame(filtered_item_profiles)
    
#     return user_profile, item_profile






def prepare_data(train, val, test, user_pro, item_pro, aspects):
    train = train[['user_enc', 'item_enc', 'rating']]
    val = val[['user_enc', 'item_enc', 'rating']]
    test = test[['user_enc', 'item_enc', 'rating']]
    user_profile = dict(zip(user_pro['user_enc'], user_pro[aspects].values))
    item_profile = dict(zip(item_pro['item_enc'], item_pro[aspects].values))
    return train, val, test, user_pro, item_pro



class RatingDataset(Dataset):
    def __init__(self, data, user_profiles, item_profiles):
        self.data = data.reset_index(drop=True)  # Ensure correct indexing
        self.user_profiles = user_profiles.set_index('user_enc')
        self.item_profiles = item_profiles.set_index('item_enc')

        # Store user IDs, item IDs, and ratings
        self.users = self.data['user_enc'].values
        self.items = self.data['item_enc'].values
        self.ratings = self.data['rating'].values  # 🔥 Fix: Define `self.ratings`

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.users[idx]  # Get user ID
        item_id = self.items[idx]  # Get item ID
        rating = self.ratings[idx]  # Get rating

        # Fetch user and item profiles
        user_profile = torch.tensor(self.user_profiles.loc[user_id].values, dtype=torch.float32)
        item_profile = torch.tensor(self.item_profiles.loc[item_id].values, dtype=torch.float32)

        return user_profile, item_profile, torch.tensor(rating, dtype=torch.float32), user_id, item_id



import numpy as np


# def ndcg_at_k(predictions, ground_truth, k=10):
#     k = min(k, len(predictions))
#     predictions = np.array(predictions)
#     ground_truth = np.array(ground_truth)

#     # Get top-k indices based on predictions
#     sorted_indices = np.argsort(-predictions)[:k]
#     sorted_ground_truth = ground_truth[sorted_indices]

#     # Compute DCG with linear gain
#     discounts = np.log2(np.arange(2, k + 2))
#     dcg = np.sum(sorted_ground_truth / discounts)

#     # Compute IDCG (ideal ordering)
#     ideal_sorted = np.sort(ground_truth)[::-1][:k]
#     idcg = np.sum(ideal_sorted / discounts)

#     # Debug prints
#     # print(f"Sorted Indices (by predictions): {sorted_indices}")
#     # print(f"Predicted Order Ground Truth: {sorted_ground_truth}")
#     # print(f"Ideal Order Ground Truth: {ideal_sorted}")
#     # print(f"DCG: {dcg:.4f}, IDCG: {idcg:.4f}, NDCG: {dcg/idcg:.4f}\n")

#     return dcg / idcg if idcg > 0 else 0.0

# def compute_ndcg(model, X_full, y_full, usr_idx, k=10):
#     """
#     Compute NDCG@k over the full dataset, ensuring per-user evaluation.

#     Args:
#         model (tf.keras.Model): Trained model.
#         X_full (tuple): Tuple containing (user_features, item_features).
#         y_full (np.array): Ground truth ratings.
#         usr_idx (np.array): User indices.
#         k (int): Rank position cutoff.

#     Returns:
#         float: Mean NDCG@k
#     """
#     # Predict ratings
#     predictions = model.predict(X_full, batch_size=128).flatten()

#     unique_users = np.unique(usr_idx)
#     total_ndcg, count = 0.0, 0

#     for user in unique_users:
#         # Mask all items rated by this user
#         user_mask = (usr_idx == user)
#         user_preds = predictions[user_mask]
#         user_truth = y_full[user_mask]

#         # Compute NDCG only if user rated more than one item
#         if len(user_truth) > 1:
#             ndcg_score = ndcg_at_k(user_preds, user_truth, k)
#             total_ndcg += ndcg_score
#             count += 1

#     return total_ndcg / count if count > 0 else 0.0  # Avoid division by zero


# def ndcg_at_k(predictions, ground_truth, k=10):
#     """
#     Compute NDCG at K for unnormalized ratings.

#     Args:
#         predictions (np.array): Predicted scores for all items.
#         ground_truth (np.array): True ratings for all items.
#         k (int): Rank position cutoff.

#     Returns:
#         float: Normalized Discounted Cumulative Gain (NDCG)
#     """
#     k = min(k, len(predictions))

#     # Sort indices based on predictions (descending order)
#     sorted_indices = np.argsort(-predictions)[:k]
#     sorted_ground_truth = ground_truth[sorted_indices]

#     # Compute DCG (Discounted Cumulative Gain)
#     dcg = np.sum((2 ** sorted_ground_truth - 1) / np.log2(np.arange(2, k + 2)))

#     # Compute IDCG (Ideal DCG)
#     ideal_sorted = np.sort(ground_truth)[::-1][:k]  # Best possible ranking
#     idcg = np.sum((2 ** ideal_sorted - 1) / np.log2(np.arange(2, k + 2)))

#     return dcg / idcg if idcg > 0 else 0.0  # Avoid division by zero

# def compute_ndcg(model, X_full, y_full, usr_idx, k=10):
#     """
#     Compute NDCG@k over the full dataset, ensuring per-user evaluation.

#     Args:
#         model (tf.keras.Model): Trained model.
#         X_full (tuple): Tuple containing (user_features, item_features).
#         y_full (np.array): Ground truth ratings.
#         usr_idx (np.array): User indices.
#         k (int): Rank position cutoff.

#     Returns:
#         float: Mean NDCG@k
#     """
#     # Predict ratings
#     predictions = model.predict(X_full, batch_size=128).flatten()

#     # Ensure usr_idx is used correctly
#     unique_users = np.unique(usr_idx)
#     total_ndcg, count = 0.0, 0

#     for user in unique_users:
#         # Mask all items rated by the user
#         user_mask = (usr_idx == user)
#         user_preds = predictions[user_mask]
#         user_truth = y_full[user_mask]

#         # Compute NDCG only if user rated more than one item
#         if len(user_truth) > 1:
#             ndcg_score = ndcg_at_k(user_preds, user_truth, k)
#             total_ndcg += ndcg_score
#             count += 1

#     return total_ndcg / count if count > 0 else 0.0  # Avoid division by zero

#for each user)
import numpy as np

def ndcg_at_k(predictions, ground_truth, k=10):
    """
    Compute NDCG@K for a single user.
    
    Args:
        predictions (np.array): Predicted scores.
        ground_truth (np.array): Actual ground truth relevance scores.
        k (int): Rank position cutoff.

    Returns:
        float: NDCG@K score.
    """
    k = min(k, len(predictions))
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Sort items by predicted scores (descending)
    sorted_indices = np.argsort(-predictions)[:k]
    sorted_ground_truth = ground_truth[sorted_indices]

    # Compute DCG
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(sorted_ground_truth / discounts)

    # Compute IDCG (ideal DCG with best possible ranking)
    ideal_sorted = np.sort(ground_truth)[::-1][:k]
    idcg = np.sum(ideal_sorted / discounts)

    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg(model, X_full, y_full, usr_idx, k=10):
    """
    Compute mean NDCG@K over all users.

    Args:
        model (tf.keras.Model): Trained model.
        X_full (tuple): User and item features.
        y_full (np.array): Ground truth ratings.
        usr_idx (np.array): User indices.
        k (int): Rank position cutoff.

    Returns:
        float: Mean NDCG@K across all users.
    """
    # Predict ratings for all items
    predictions = model.predict(X_full, batch_size=128).flatten()

    unique_users = np.unique(usr_idx)
    ndcg_scores = []

    for user in unique_users:
        # Get all items interacted with by this user
        user_mask = (usr_idx == user)
        user_preds = predictions[user_mask]
        user_truth = y_full[user_mask]

        # Compute NDCG only if the user has rated at least 2 items
        if len(user_truth) > 1:
            ndcg_score = ndcg_at_k(user_preds, user_truth, k)
            ndcg_scores.append(ndcg_score)

    # Compute mean NDCG across users (avoid division by zero)
    return np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0

import numpy as np

def compute_ndcg_raw(model, X_test, y_test, k=10):
    """
    Compute NDCG@K for the entire test set.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (tuple): Test user and item features.
        y_test (np.array): Ground truth ratings.
        k (int): Rank position cutoff.

    Returns:
        float: NDCG@K score for the entire dataset.
    """
    # Predict ratings for all test samples
    predictions = model.predict(X_test, batch_size=128).flatten()
    
    # Ensure K does not exceed dataset size
    k = min(k, len(predictions))
    
    # Sort all items by predicted scores (descending)
    sorted_indices = np.argsort(-predictions)[:k]
    sorted_ground_truth = y_test[sorted_indices]

    # Compute DCG
    discounts = np.log2(np.arange(2, k + 2))  # Discount factor
    dcg = np.sum(sorted_ground_truth / discounts)

    # Compute IDCG (Ideal DCG with the best possible ranking)
    ideal_sorted = np.sort(y_test)[::-1][:k]
    idcg = np.sum(ideal_sorted / discounts)

    return dcg / idcg if idcg > 0 else 0.0


def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)))




def plot_training_history_and_model(history, model):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='green')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    if 'rmse' in history.history:
        plt.plot(history.history['rmse'], label='Train RMSE')
        plt.plot(history.history['val_rmse'], label='Val RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
    else:
        print("RMSE metrics not found in history.")
    plt.tight_layout()
    plt.show()
    print("Model Architecture:")
    model.summary()





def prepare_tf_data(data_loader, concatenate=False):
    user_profiles, item_profiles, ratings, usr_idx, item_idx = [], [], [], [], []
    
    for user_profile_batch, item_profile_batch, rating_batch, users, items in data_loader:
        user_profiles.append(user_profile_batch.numpy())
        item_profiles.append(item_profile_batch.numpy())
        ratings.append(rating_batch.numpy())
        usr_idx.append(users.numpy())  # Append user indices
        item_idx.append(items.numpy())  # Append item indices

    user_profiles = np.vstack(user_profiles)
    item_profiles = np.vstack(item_profiles)
    ratings = np.concatenate(ratings)
    usr_idx = np.concatenate(usr_idx)  # Convert list to array
    item_idx = np.concatenate(item_idx)

    if concatenate:
        return np.hstack([user_profiles, item_profiles]), ratings, usr_idx, item_idx
    else:
        return [user_profiles, item_profiles], ratings, usr_idx, item_idx


# second:
# def prepare_tf_data(data_loader, concatenate=False):
#     user_profiles, item_profiles, ratings, usr_idx, item_idx = [], [], [], [], []
    
#     for user_profile_batch, item_profile_batch, rating_batch, users, items in data_loader:
#         user_profiles.append(user_profile_batch.numpy())
#         item_profiles.append(item_profile_batch.numpy())
#         ratings.append(rating_batch.numpy())
#         usr_idx.append(users.numpy())  # Append user indices
#         item_idx.append(items.numpy())  # Append item indices

#     user_profiles = np.vstack(user_profiles)
#     item_profiles = np.vstack(item_profiles)
#     ratings = np.concatenate(ratings)
#     usr_idx = np.concatenate(usr_idx)  # Convert list to array
#     item_idx = np.concatenate(item_idx)

#     return [usr_idx, item_idx, user_profiles, item_profiles], ratings




import random
import numpy as np

def recommend_items_full(model, X_full, y_full, usr_idx, itm_idx, num_recommendations=10):
    """
    Recommend items to a user in the full dataset (validation + test) and show predicted vs actual ratings.

    Args:
        model (tf.keras.Model): Trained recommendation model.
        X_full (tuple): Tuple ([user_profiles, item_profiles]) containing full dataset.
        y_full (np.array): Actual ratings.
        usr_idx (np.array): User indices in full dataset.
        itm_idx (np.array): Item indices in full dataset.
        num_recommendations (int): Number of items to recommend.

    Returns:
        None
    """
    # Select a user that actually exists in the dataset
    unique_users = np.unique(usr_idx)
    selected_user = random.choice(unique_users)  # Pick a random user from the dataset

    # Find indices where this user appears in the full dataset
    user_mask = usr_idx == selected_user

    # Get their corresponding item embeddings and actual ratings
    item_embeddings = X_full[1][user_mask]  # Extract corresponding item embeddings
    actual_ratings = y_full[user_mask]  # Extract corresponding actual ratings
    item_ids = itm_idx[user_mask]  # Extract corresponding item IDs

    # If no ratings are found, skip to the next user
    if len(actual_ratings) == 0:
        print(f"No interactions found for User ID {selected_user}. Trying another user...")
        return recommend_items_full(model, X_full, y_full, usr_idx, itm_idx, num_recommendations)  # Retry with another user

    # Get the user profile corresponding to the selected user (it will be the same for all items)
    user_embedding = X_full[0][user_mask][0]  # Take any user profile from the user_mask, since it will be the same for all items

    # Repeat the user embedding for all items the user has interacted with
    repeated_user_embedding = np.tile(user_embedding, (len(item_embeddings), 1))

    # Predict ratings using the model (input user embeddings and item embeddings)
    predicted_ratings = model.predict([repeated_user_embedding, item_embeddings]).flatten()

    # Create a list of recommendations with item ID, predicted rating, and actual rating
    recommendations = list(zip(item_ids, predicted_ratings, actual_ratings))
    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating

    # Display recommendations
    print(f"\nRecommendations for User ID {selected_user}:")
    print(f"{'Item ID':<10}{'Predicted Rating':<20}{'Actual Rating':<15}")
    for item_id, predicted, actual in recommendations[:num_recommendations]:
        print(f"{item_id:<10}{predicted:<20.4f}{actual:<15.4f}")

    return recommendations

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluation_auc(model, X_test, y_test, threshold=3):
    """
    Evaluates the model and calculates binary classification metrics.
    
    Parameters:
    - model: Trained TensorFlow/Keras model
    - X_test: Test features
    - y_test: True labels (ratings)
    - threshold: The threshold for binarizing ratings (default=3.5)
    
    Returns:
    - Dictionary containing accuracy, AUC, micro/macro F1-score.
    """
    
    # Get model predictions
    y_pred = model.predict(X_test)
    
    # Convert continuous predictions to binary labels
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    auc = roc_auc_score(y_test_binary, y_pred)  # Use raw predictions for AUC
    f1_micro = f1_score(y_test_binary, y_pred_binary, average='micro')
    f1_macro = f1_score(y_test_binary, y_pred_binary, average='macro')
    f1_weighted = f1_score(y_test_binary, y_pred_binary, average='weighted')

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }



import os
import random
import numpy as np
from datetime import datetime

import os
import random
import numpy as np
from datetime import datetime

def explainability(model, X_full, y_full, usr_idx, itm_idx, 
                          user_profiles, item_profiles, aspects, 
                          user_col="user_enc", item_col="item_enc",
                          num_recommendations=10, explanation_path="explanation.txt",
                          threshold_std=0.5):
    """
    1) Select a random user from the dataset.
    2) Predict and sort items by rating for that user.
    3) Identify the user's important aspects (high or low sentiment).
    4) Provide a concise explanation of how each recommended item aligns with those aspects.
    5) Write the explanation to a text file and return it along with the recommendations.
    
    Args:
        model (tf.keras.Model): Trained recommendation model.
        X_full (tuple): Tuple ([user_profiles, item_profiles]) containing the full dataset features.
        y_full (np.array): Actual ratings corresponding to X_full.
        usr_idx (np.array): User indices in the dataset.
        itm_idx (np.array): Item indices in the dataset.
        user_profiles (pd.DataFrame): DataFrame mapping user IDs to aspect sentiment scores.
        item_profiles (pd.DataFrame): DataFrame mapping item IDs to aspect sentiment scores.
        aspects (list): List of aspect names.
        user_col (str): Column name in user_profiles for user IDs.
        item_col (str): Column name in item_profiles for item IDs.
        num_recommendations (int): Number of items to recommend.
        explanation_path (str): Path to the file where the explanation is saved.
        threshold_std (float): Standard deviation multiplier for deciding which aspects 
                               are important (either high or low).
    
    Returns:
        (explanation_text, top_recommendations):
            - explanation_text: A string containing the explanation.
            - top_recommendations: List of tuples (item_id, predicted_rating, actual_rating).
    """
    # 1. Select a random user
    unique_users = np.unique(usr_idx)
    selected_user = random.choice(unique_users)
    user_mask = usr_idx == selected_user
    
    # Extract item embeddings, actual ratings, and item IDs for this user
    item_embeddings = X_full[1][user_mask]
    actual_ratings = y_full[user_mask]
    item_ids = itm_idx[user_mask]
    
    if len(actual_ratings) == 0:
        print(f"No interactions found for User ID {selected_user}. Trying another user...")
        return explainability(model, X_full, y_full, usr_idx, itm_idx, 
                                     user_profiles, item_profiles, aspects, 
                                     user_col, item_col, num_recommendations, 
                                     explanation_path, threshold_std)
    
    # 2. Predict ratings for all items this user has interacted with
    user_embedding = X_full[0][user_mask][0]
    repeated_user_embedding = np.tile(user_embedding, (len(item_embeddings), 1))
    predicted_ratings = model.predict([repeated_user_embedding, item_embeddings]).flatten()
    
    # Sort items by predicted rating
    recommendations = list(zip(item_ids, predicted_ratings, actual_ratings))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:num_recommendations]
    
    # 3. Identify the user's important aspects
    #    (global average & std come from user_profiles across all users)
    user_aspect_avgs = {aspect: user_profiles[aspect].mean() for aspect in aspects}
    user_aspect_stds = {aspect: user_profiles[aspect].std() for aspect in aspects}
    
    user_profile_row = user_profiles[user_profiles[user_col] == selected_user]
    user_profile_dict = (user_profile_row.iloc[0].to_dict() if not user_profile_row.empty else {})
    
    important_aspects = {}
    for aspect in aspects:
        if aspect not in user_profile_dict:
            continue
        avg = user_aspect_avgs[aspect]
        std = user_aspect_stds[aspect]
        val = user_profile_dict[aspect]
        high_threshold = avg + threshold_std * std
        low_threshold  = avg - threshold_std * std
        # If user sentiment is significantly above or below average, mark as important
        if val > high_threshold or val < low_threshold:
            important_aspects[aspect] = val
    
    # 4. Build a concise explanation
    lines = []
    lines.append(f"Explainability Report for User ID: {selected_user}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    lines.append("Important Aspects (high or low):")
    if important_aspects:
        for aspect, user_val in important_aspects.items():
            lines.append(f"  {aspect}: {user_val:.4f}")
    else:
        lines.append("  (None identified based on threshold)")
    lines.append("")
    
    lines.append("Top Recommended Items:")
    for item_id, pred, actual in top_recommendations:
        lines.append(f"Item ID: {item_id}")
        lines.append(f"  Predicted Rating: {pred:.4f} | Actual Rating: {actual:.4f}")
        
        # Show the relevant aspect matches if available
        item_profile_row = item_profiles[item_profiles[item_col] == item_id]
        if not item_profile_row.empty:
            item_profile_dict = item_profile_row.iloc[0].to_dict()
            # For each important aspect, display the item's corresponding score
            for aspect, user_val in important_aspects.items():
                item_val = item_profile_dict.get(aspect, None)
                if item_val is not None:
                    lines.append(f"    {aspect}: User = {user_val:.4f}, Item = {item_val:.4f}")
        lines.append("")
    
    explanation_text = "\n".join(lines)
    
    # 5. Write explanation to file
    mode = "a" if os.path.exists(explanation_path) else "w"
    with open(explanation_path, mode, encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(explanation_text + "\n")
    
    print(f"Explainability report saved to {explanation_path}")
    return explanation_text, top_recommendations
