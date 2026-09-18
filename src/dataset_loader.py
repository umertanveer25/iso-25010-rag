"""
Dataset Loader for the 14-class FNFC Software Requirements Dataset.
Handles robust file loading, text normalization, and stratified splitting.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_DATASET_PATH = r"C:\Users\umert\Downloads\FNFC.csv"

# Standard 14 FNFC Class Order
FNFC_CLASSES = ["F", "SE", "AU", "P", "A", "PE", "US", "O", "LL", "R", "M", "LF", "FT", "SC"]

def load_fnfc_dataset(filepath=DEFAULT_DATASET_PATH, test_size=0.2, random_state=42):
    """
    Loads FNFC dataset, cleans text, encodes labels, and creates stratified train/test split.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Load with robust encoding fallback
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    # Normalize columns
    df.columns = [col.strip().lower() for col in df.columns]
    if 'text' not in df.columns or 'class' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'class' columns.")

    # Clean text
    df['clean_text'] = df['text'].astype(str).str.strip()
    df['class'] = df['class'].astype(str).str.strip().str.upper()

    # Filter any unexpected classes
    valid_df = df[df['class'].isin(FNFC_CLASSES)].copy()

    # Fit Label Encoder
    encoder = LabelEncoder()
    encoder.fit(FNFC_CLASSES)
    valid_df['label'] = encoder.transform(valid_df['class'])

    # Stratified Train/Test Split
    train_df, test_df = train_test_split(
        valid_df,
        test_size=test_size,
        random_state=random_state,
        stratify=valid_df['label']
    )

    print(f"Loaded {len(valid_df)} requirement statements across {len(encoder.classes_)} classes.")
    print(f"Train split: {len(train_df)} samples | Test split: {len(test_df)} samples")
    
    return train_df, test_df, encoder, valid_df

if __name__ == "__main__":
    train_df, test_df, encoder, full_df = load_fnfc_dataset()
    print("Class Counts in Full Dataset:\n", full_df['class'].value_counts())
