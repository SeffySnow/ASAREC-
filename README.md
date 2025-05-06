
# ASAREC‑Pipeline

A lightweight pipeline for **Aspect‑Sentiment Aware Recommender System** using DeBERTa‑based ABSA and explicit interaction modeling.


## 🏗️ Model Architecture

   This repository implements an **Aspect‑Sentiment‑Aware Recommender System**, combining the fine‑grained insights of aspect‑based analysis with the polarity information users express toward those aspects. 
   
   Reviews are a rich source of customer opinions, people naturally comment on the specific features they care most about. By modeling both *what* aspects matter and *how* customers feel about them, our approach not only yields more accurate recommendations but also provides clear explanations for why a user may like or dislike an item, and reveals public sentiment trends across products.

The pipeline consists of three main stages:

1. **Aspect‑Based Sentiment Extraction**  
   We use a DeBERTa‑v3 ABSA model to score each review on a predefined set of domain‑specific aspects, producing a sentiment polarity vector for every review.

2. **User & Item Profile Construction**  
   Sentiment vectors are averaged across all reviews by each user and all reviews for each item, generating compact “preference” and “characteristic” profiles.

3. **Interaction‑Based Rating Prediction**  
   We compute explicit compatibility features—element‑wise product, difference, and cosine similarity—between user and item profiles, and feed these into a neural predictor to estimate ratings.


---

## 📂 Repository Structure

1. Dowload the datasets.
#Datasets: 
* https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
* https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz

2. Installing the dependencies :

  pip install -r requirements.txt

3.  Preprocessing (filter, encode & split)

4. ABSA extraction (aspect–sentiment scoring), feel free to modify the set of aspects

5. Model training (profile construction, feature engineering & MLP)

6. The results will be saved in datasets/"dataset_name"/results.json



