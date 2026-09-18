"""
Angle 1: ISO 25010 Knowledge-Augmented RAG Attention Fusion Classifier for FNFC Dataset.
Uses Gated Attention Fusion with MaxAbs Scaling for ultra-fast and accurate execution.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from iso_rag_retriever import ISORagRetriever

class FNFCBaselineClassifier:
    """
    Standard TF-IDF Baseline Classifier (no ISO knowledge augmentation).
    """
    def __init__(self, max_features=5000):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), sublinear_tf=True, stop_words='english')
        self.clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=2.0, random_state=42)

    def fit(self, train_texts, train_labels):
        X_train = self.tfidf.fit_transform(train_texts)
        self.clf.fit(X_train, train_labels)
        return self

    def predict(self, test_texts):
        X_test = self.tfidf.transform(test_texts)
        return self.clf.predict(X_test)

    def predict_proba(self, test_texts):
        X_test = self.tfidf.transform(test_texts)
        return self.clf.predict_proba(X_test)


class ISOAugmentedFusionClassifier:
    """
    Angle 1: ISO 25010 Gated Attention RAG Fusion Classifier.
    Fuses Text Features with ISO 25010 RAG Similarity Context using Gated Attention.
    """
    def __init__(self, fnfc_classes, max_features=5000, iso_scale=3.0):
        self.fnfc_classes = fnfc_classes
        self.retriever = ISORagRetriever(fnfc_classes)
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), sublinear_tf=True, stop_words='english')
        self.scaler = MaxAbsScaler()
        self.iso_scale = iso_scale
        
        # High-performance Regularized Multi-class Classifier
        self.clf = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            C=3.5, 
            random_state=42
        )

    def _extract_fusion_features(self, texts, is_train=False):
        # 1. Text TF-IDF N-grams
        if is_train:
            text_sparse = self.tfidf.fit_transform(texts)
        else:
            text_sparse = self.tfidf.transform(texts)

        # 2. ISO 25010 RAG context features (28D)
        iso_feats = self.retriever.transform_all_features(texts) # Shape: (N, 28)
        
        # 3. Gated Attention Feature Construction
        max_sim = np.max(iso_feats[:, :14], axis=1, keepdims=True) # (N, 1)
        mean_sim = np.mean(iso_feats[:, :14], axis=1, keepdims=True) # (N, 1)
        
        # Sigmoid gate over maximum ISO similarity
        gate = 1.0 / (1.0 + np.exp(-4.0 * (max_sim - 0.12)))
        gated_iso = iso_feats * gate * self.iso_scale
        
        # Combine gated ISO features and similarity signals
        augmented_iso = np.hstack([gated_iso, max_sim * 2.0, mean_sim])
        iso_sparse = sp.csr_matrix(augmented_iso)
        
        fused_sparse = sp.hstack([text_sparse, iso_sparse])
        
        if is_train:
            return self.scaler.fit_transform(fused_sparse)
        else:
            return self.scaler.transform(fused_sparse)

    def fit(self, train_texts, train_labels):
        X_train = self._extract_fusion_features(train_texts, is_train=True)
        self.clf.fit(X_train, train_labels)
        return self

    def predict(self, test_texts):
        X_test = self._extract_fusion_features(test_texts, is_train=False)
        return self.clf.predict(X_test)

    def predict_proba(self, test_texts):
        X_test = self._extract_fusion_features(test_texts, is_train=False)
        return self.clf.predict_proba(X_test)
