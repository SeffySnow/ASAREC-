
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import json
from utils import *
import matplotlib as plt 
from torch.utils.data import ConcatDataset, DataLoader
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.path.exists("base_w_6.keras"):
    os.remove("base_w_6.keras")  # Remove old model
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))



tf.keras.backend.set_floatx('float32')
# print(f"TensorFlow: {tf.__version__}")
# print("GPU Available:", "GPU" in [d.device_type for d in tf.config.list_physical_devices()])







layers = keras.layers
Model = keras.Model
regularizers = keras.regularizers
ModelCheckpoint = keras.callbacks.ModelCheckpoint


# MAIN
def interaction_model(num_aspects,learning_rate):
    """
    Model 6: Predict ratings using enhanced interactions (element-wise + cosine similarity).
    Adds an extra dense layer for deeper modeling.

    Args:
        num_aspects (int): Number of aspects (dimensions of embeddings).

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    # Inputs
    user_input = layers.Input(shape=(num_aspects,), name="user_input")
    item_input = layers.Input(shape=(num_aspects,), name="item_input")

    # Interaction Features
    interaction_mult = layers.Multiply()([user_input, item_input])  # Element-wise multiplication
    interaction_diff = layers.Subtract()([user_input, item_input])  # Element-wise subtraction

    # Cosine Similarity
    cosine_sim = layers.Dot(axes=1, normalize=True)([user_input, item_input])
    cosine_sim_expanded = layers.Reshape((1,))(cosine_sim)  #

    # Combine All Interaction Features
    interaction_concat = layers.Concatenate()([interaction_mult, interaction_diff,cosine_sim_expanded])
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Layer

    output = layers.Dense(1,activation = "linear")(x)

    # Model
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # loss=tf.keras.losses.Huber(delta=1.0),
        loss='mean_squared_error',
        metrics=['mae', 'mse', rmse]
    )
    return model
#----------------------------------------




def save_results(dataset_name, results, epochs, learning_rate, filename="results.json"):
    """
    Save training results to a JSON file. If the file exists, append to it.

    Args:
        dataset_name (str): Name of the dataset.
        results (dict): Dictionary containing the results (loss, MAE, RMSE).
        epochs (int): Number of epochs used for training.
        learning_rate (float): Learning rate used for training.
        filename (str): Name of the JSON file to store results.
    """
    new_entry = {
        "dataset_name": dataset_name,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "loss": results[0],
        "mae": results[1],
        "mse": results[2],
        "rmse": results[3]
    }

    # Load existing data if the file exists
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append(new_entry)

    # Save back to file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Results saved to {filename}")

def run_stepwise_selection(user_pro, item_pro, train, val, test, aspects, method="forward"):
    """
    Perform forward or backward step selection to find important aspects.
    """
    from copy import deepcopy

    all_aspects = aspects.copy()
    selected_aspects = [] if method == "forward" else aspects.copy()
    performance_history = []

    print(f"\nStarting {method} step selection...\n")

    while True:
        best_score = float('inf')
        best_aspect = None
        best_metrics = None

        candidates = (
            [a for a in all_aspects if a not in selected_aspects] if method == "forward"
            else selected_aspects
        )

        if not candidates:
            break

        for aspect in candidates:
            if method == "forward":
                temp_aspects = selected_aspects + [aspect]
            else:  # backward
                temp_aspects = [a for a in selected_aspects if a != aspect]

            # ✅ Keep 'user_enc' and 'item_enc' when subsetting
            user_subset = user_pro[['user_enc'] + temp_aspects]
            item_subset = item_pro[['item_enc'] + temp_aspects]

            # Prepare datasets
            train_sub, val_sub, test_sub, user_sub, item_sub = prepare_data(
                deepcopy(train), deepcopy(val), deepcopy(test), user_subset, item_subset, temp_aspects
            )
            train_loader = DataLoader(RatingDataset(train_sub, user_sub, item_sub), batch_size=128, shuffle=True)
            val_loader = DataLoader(RatingDataset(val_sub, user_sub, item_sub), batch_size=128, shuffle=False)

            X_train, y_train, _, _ = prepare_tf_data(train_loader, concatenate=False)
            X_val, y_val, _, _ = prepare_tf_data(val_loader, concatenate=False)

            # Train model
            model = interaction_model(num_aspects=len(temp_aspects), learning_rate=0.0005)
            model.fit(X_train, y_train, epochs=15, batch_size=128, verbose=0)

            results = model.evaluate(X_val, y_val, verbose=0)
            val_mae, val_mse = results[1], results[2]

            if val_mae < best_score:
                best_score = val_mae
                best_aspect = aspect
                best_metrics = {"mae": val_mae, "mse": val_mse}

        if method == "forward":
            if best_aspect:
                selected_aspects.append(best_aspect)
        else:
            selected_aspects.remove(best_aspect)

        performance_history.append((deepcopy(selected_aspects), best_metrics))

        print(f"Selected Aspects: {selected_aspects}")
        print(f"MAE: {best_metrics['mae']:.4f}, MSE: {best_metrics['mse']:.4f}\n")

        if method == "backward" and len(selected_aspects) == 1:
            break

    return performance_history
import matplotlib.pyplot as plt

def plot_results(performance_history, method):
    steps = list(range(1, len(performance_history)+1))
    maes = [m['mae'] for _, m in performance_history]
    mses = [m['mse'] for _, m in performance_history]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, maes, label='MAE')
    plt.plot(steps, mses, label='MSE')
    plt.xlabel("Number of Selected Aspects")
    plt.ylabel("Validation Error")
    plt.title(f"{method.capitalize()} Stepwise Selection")
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    from datetime import datetime

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # Remove existing model file if it exists
    best_model_path = "base_w_6.keras"
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    # Rest of your existing code...

    parser = argparse.ArgumentParser(description="Train the recommendation model")
    parser.add_argument('dataset_name', type=str, help="Dataset name (e.g., movie, book)",default="movie")
    parser.add_argument('abbreviation', type=str, help="Dataset abbreviation (e.g., mv, bk)",default="mv")
    parser.add_argument('-e', '--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    print("importing datasets")

  
    dataset_path = f"dataset/{args.dataset_name}/"
    review_path = f"{dataset_path}review_{args.abbreviation}.csv"
    train_path = f"{dataset_path}train_{args.abbreviation}.csv"
    val_path = f"{dataset_path}val_{args.abbreviation}.csv"
    test_path = f"{dataset_path}test_{args.abbreviation}.csv"
    aspects_path = f"{dataset_path}aspects.txt"

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    review = pd.read_csv(review_path)
    print(train['user_enc'].nunique(), train['item_enc'].nunique())
    print(test.shape)
    with open(aspects_path, 'r',encoding="utf-8") as file:
        aspects = [line.strip().strip('",') for line in file if line.strip()]

    item_col = "item_enc"
    user_col = "user_enc"
    

    print("creating profiles")
    user_pro , item_pro = profiles(review, item_col, user_col, aspects) 
    print("user profile shape:", user_pro.shape)
    print("item profile shape:", item_pro.shape)
    # # Run Forward Selection
    # forward_results = run_stepwise_selection(user_pro, item_pro, train, val, test, aspects, method="forward")

    # # Run Backward Elimination
    # backward_results = run_stepwise_selection(user_pro, item_pro, train, val, test, aspects, method="backward")
    # plot_results(forward_results, "forward")
    # plot_results(backward_results, "backward")

    # # Debug: Check if there are missing item IDs
    # missing_items = set(train[item_col].unique()) - set(item_pro.index)

    # print(f"Total items in dataset: {train[item_col].nunique()}")
    # print(f"Total items in item_profile: {item_pro.shape[0]}")
    # print(f"Missing items in item_profile: {len(missing_items)}")

    # if missing_items:
    #     print("Missing Item IDs (First 10):", list(missing_items)[:10])  # Print a few missing IDs

    train, val, test, user_pro, item_pro = prepare_data(train, val, test, user_pro, item_pro, aspects)


    print("preparing data...\n")
    train_dataset = RatingDataset(train, user_pro, item_pro)
    val_dataset = RatingDataset(val, user_pro, item_pro)
    test_dataset = RatingDataset(test, user_pro, item_pro)

    train_loader =  DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, 
    shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    user_profile_batch, item_profile_batch, rating_batch ,_,_= next(iter(train_loader))
    input_shape = user_profile_batch.shape[1]
    print("Input Shape:", input_shape)


    print("model training start")
    model_6 = interaction_model(input_shape, learning_rate=learning_rate)
    model_6.summary()
    best_model_path = "base_w_6.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    X_train, y_train ,_, _= prepare_tf_data(train_loader, concatenate=False)
    X_val, y_val ,_, _= prepare_tf_data(val_loader, concatenate=False)
    X_test, y_test,_, _ = prepare_tf_data(test_loader, concatenate=False)

    model_6.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
        epochs=epochs,
        batch_size=64
    )

    best_model_6 = tf.keras.models.load_model(best_model_path, custom_objects={'rmse': rmse})
    
    # -----------------------------------------------------------------------------------------------------

    start_time = time.time()

# # Evaluate the model
    results = best_model_6.evaluate(X_test, y_test)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Test Results - Loss: {results[0]}, MAE: {results[1]}, MSE {results[2]}, RMSE: {results[3]}")
    print(f"Evaluation Time: {elapsed_time:.4f} seconds")


    # -----------------------------------------------------------------------------------------------------

    print("\n\n\n---------------------------------------")
    print("\n\nCalculating recommendation metrics ... \n")

    X_test, y_test , usr_idx, itm_idx  = prepare_tf_data(test_loader, concatenate=False)
    ndcg_score = compute_ndcg_raw(best_model_6, X_test, y_test, k=10)


    print(f"Test Results - Loss: {results[0]}, MAE: {results[1]}, MSE: {results[2]}, RMSE: {results[3]}, NDCG@k : {ndcg_score}")
    print(recommend_items_full(best_model_6, X_test , y_test, usr_idx, itm_idx, num_recommendations=10))
    auc_test = evaluation_auc(best_model_6, X_test, y_test, threshold=3.5)
    print(auc_test)
    # Save to JSON
    results_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "hyperparameters": vars(args),
        "metrics": {
            "loss": float(results[0]),
            "mae": float(results[1]),
            "mse": float(results[2]),
            "ndcg@10": float(ndcg_score)
        }
    }

    results_dir = f"dataset/{args.dataset_name}/"
    results_file = os.path.join(results_dir, "results.json")
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_data = json.load(f)
    else:
        results_data = []

    results_data.append(results_entry)
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"\nResults saved to {results_file}")





  

 

if __name__ == "__main__":
    main()







