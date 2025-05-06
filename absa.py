import os
import argparse
import pandas as pd
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Patterns for cleaning
HTML_PATTERN = re.compile(r'<.*?>')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s]')

# Text cleaner
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = HTML_PATTERN.sub('', text)
    return SPECIAL_CHARS_PATTERN.sub('', text).strip()

# Batched aspect sentiment
def analyze_aspect_sentiment_batched(sentences, aspects, tokenizer, model, device, max_length=128):
    inputs = tokenizer(
        [f"[CLS] {s} [SEP] {a} [SEP]" for s, a in zip(sentences, aspects)],
        return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float()
        probs = F.softmax(logits, dim=1)
    return probs[:, 2].cpu().numpy()

# Full review analysis
def analyze_reviews(df, aspect_list, tokenizer, model, device, batch_size, sub_batch_size):
    df = df.copy()
    df['cleaned_text'] = df['text'].astype(str).map(clean_text)
    results = {}
    for i in tqdm(range(0, len(df), batch_size), desc="Batches"):
        batch = df.iloc[i:i+batch_size]
        aspect_scores = {asp: [] for asp in aspect_list}
        for asp in aspect_list:
            texts = batch['cleaned_text'].tolist()
            aspects = [asp] * len(texts)
            scores = []
            for j in range(0, len(texts), sub_batch_size):
                sub_texts = texts[j:j+sub_batch_size]
                sub_aspects = aspects[j:j+sub_batch_size]
                s = analyze_aspect_sentiment_batched(sub_texts, sub_aspects, tokenizer, model, device)
                scores.extend(s)
                torch.cuda.empty_cache()
            aspect_scores[asp] = scores
        for idx, orig_idx in enumerate(batch.index):
            results[orig_idx] = {asp: aspect_scores[asp][idx] for asp in aspect_list}
    return results

# Main extraction function
def extract_aspect_sentiments(folder, abbr, batch_size=32, sub_batch_size=16):
    data_dir = os.path.join('data', folder)
    # Load aspects from aspects.txt
    aspects_file = os.path.join(data_dir, 'aspects.txt')
    with open(aspects_file, 'r') as f:
        aspects = [line.strip() for line in f if line.strip()]

    # Load train file
    train_path = os.path.join(data_dir, f'train_{abbr}.csv')
    df = pd.read_csv(train_path)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('yangheng/deberta-v3-base-absa-v1.1')
    model = AutoModelForSequenceClassification.from_pretrained('yangheng/deberta-v3-base-absa-v1.1')
    model.to(device).eval()
    torch.set_grad_enabled(False)

    # Analyze
    scores = analyze_reviews(df, aspects, tokenizer, model, device, batch_size, sub_batch_size)
    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=aspects)

    # Merge and save
    merged = pd.concat([df[['user_enc','item_enc','rating']].reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    out_path = os.path.join(data_dir, f'review_{abbr}.csv')
    merged.to_csv(out_path, index=False)
    print(f"Saved review file: {out_path}")

# CLI
def main():
    parser = argparse.ArgumentParser(description='Extract ABSA review sentiments from aspects.txt')
    parser.add_argument('folder', type=str, help='Folder name under data/')
    parser.add_argument('abbr', type=str, help='Abbreviation for file names')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sub_batch_size', type=int, default=16)
    args = parser.parse_args()

    extract_aspect_sentiments(args.folder, args.abbr, args.batch_size, args.sub_batch_size)

if __name__ == '__main__':
    main()
