import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def process_data(data, min_user, min_item, min_review_length=5, output_dir=None, abbr=None):
    data = data[["rating", "text", "parent_asin", "user_id"]]
    print("Initial dataset statistics:")
    print(f"Total reviews: {data.shape[0]}")
    print(f"Total unique users: {data['user_id'].nunique()}")
    print(f"Total unique items: {data['parent_asin'].nunique()}")

    data = data[data['text'].str.len() >= min_review_length]
    item_counts = data['parent_asin'].value_counts()
    data = data[data['parent_asin'].isin(item_counts[item_counts >= min_item].index)]
    user_counts = data['user_id'].value_counts()
    data = data[data['user_id'].isin(user_counts[user_counts >= min_user].index)]
    item_counts = data['parent_asin'].value_counts()
    data = data[data['parent_asin'].isin(item_counts[item_counts > 1].index)]
    user_counts = data['user_id'].value_counts()
    data = data[data['user_id'].isin(user_counts[user_counts > 1].index)]

    print("\nFiltered dataset statistics:")
    print(f"Total reviews after filtering: {data.shape[0]}")
    print(f"Total unique users after filtering: {data['user_id'].nunique()}")
    print(f"Total unique items after filtering: {data['parent_asin'].nunique()}")
    print("\nRating Distribution:")
    print(data['rating'].value_counts())

    if output_dir and abbr:
        plt.figure(figsize=(8, 5))
        data['rating'].value_counts().sort_index().plot(kind='bar', alpha=0.7)
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Rating Distribution')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rating_distribution_{abbr}.png"))
        plt.close()
    return data


def preprocess_domain(df, output_dir, fn):
    df = df.sample(frac=1).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df = df.drop_duplicates().reset_index(drop=True)

    user_le = LabelEncoder()
    item_le = LabelEncoder()
    df['user_enc'] = user_le.fit_transform(df['user_id'])
    df['item_enc'] = item_le.fit_transform(df['parent_asin'])

    pd.DataFrame({'user_enc': range(len(user_le.classes_)), 'user_id': user_le.classes_}) \
        .to_csv(os.path.join(output_dir, f"{fn}_user_enc_to_id.csv"), index=False)
    pd.DataFrame({'item_enc': range(len(item_le.classes_)), 'parent_asin': item_le.classes_}) \
        .to_csv(os.path.join(output_dir, f"{fn}_item_enc_to_asin.csv"), index=False)
    return df


def leave_one_out_split_with_validation(df, output_dir, abbr, random_state=None):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    user_counts = df['user_enc'].value_counts()
    eligible_users = user_counts[user_counts > 6].index
    test_idxs, val_idxs = [], []
    for u in eligible_users:
        udata = df[df['user_enc'] == u]
        n = len(udata)
        n_test = n // 4; n_val = n // 4
        if n_test > 0 and n_val > 0:
            test = udata.sample(n=n_test, random_state=random_state)
            rem = udata.drop(test.index)
            val = rem.sample(n=n_val, random_state=random_state)
            test_idxs += list(test.index)
            val_idxs += list(val.index)
    test_df = df.loc[test_idxs].reset_index(drop=True)
    val_df = df.loc[val_idxs].reset_index(drop=True)
    train_df = df.drop(test_idxs + val_idxs).reset_index(drop=True)

    train_df['review_id'] = train_df.index
    val_df['review_id'] = val_df.index
    test_df['review_id'] = test_df.index

    train_df.to_csv(os.path.join(output_dir, f"train_{abbr}.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"val_{abbr}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test_{abbr}.csv"), index=False)
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Process and split review datasets')
    parser.add_argument('input_path', type=str, help='Path to input CSV file')
    parser.add_argument('folder_name', type=str, help='Name for output folder')
    parser.add_argument('abbr', type=str, help='Abbreviation for file names')
    parser.add_argument('--min_user', type=int, default=15)
    parser.add_argument('--min_item', type=int, default=30)
    parser.add_argument('--min_review_length', type=int, default=20)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.join('data', args.folder_name)
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(args.input_path)
    filtered = process_data(data, args.min_user, args.min_item, args.min_review_length, output_dir, args.abbr)
    filtered = filtered.drop_duplicates(subset=['user_id', 'parent_asin'])
    filtered.to_csv(os.path.join(output_dir, f"{args.folder_name}.csv"), index=False)

    df_enc = preprocess_domain(filtered, output_dir, args.folder_name)
    leave_one_out_split_with_validation(df_enc, output_dir, args.abbr, args.random_state)

    print("Done. Files in:", output_dir)

if __name__ == '__main__':
    main()