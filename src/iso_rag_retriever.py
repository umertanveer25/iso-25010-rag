
"""
Vectorized ISO 25010 Knowledge-Augmented RAG Retriever.
Uses CountVectorizer & TfidfVectorizer over 14 ISO profiles for ultra-fast feature extraction.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from iso_knowledge_base import ISO_25010_KNOWLEDGE_BASE

class ISORagRetriever:
    def __init__(self, fnfc_classes):
        self.fnfc_classes = fnfc_classes
        self.class_profiles = []
        self.kw_vocab = []

        for cls in fnfc_classes:
            info = ISO_25010_KNOWLEDGE_BASE.get(cls, {})
            name = info.get("name", "")
            iso_cat = info.get("iso_category", "")
            definition = info.get("definition", "")
            sub_chars = " ".join(info.get("sub_characteristics", []))
            keywords = " ".join(info.get("keywords", []))

            profile_text = f"{name} {iso_cat} {definition} {sub_chars} {keywords}"
            self.class_profiles.append(profile_text)
            self.kw_vocab.extend(info.get("keywords", []))

        # 1. Cosine similarity TF-IDF
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.profile_matrix = self.tfidf.fit_transform(self.class_profiles)

        # 2. Vectorized Keyword Hit Count
        self.kw_vocab = list(set([k.lower() for k in self.kw_vocab if k.strip()]))
        self.count_vec = CountVectorizer(vocabulary=self.kw_vocab)
        self.count_vec.fit(self.class_profiles)
        vocab_map = self.count_vec.vocabulary_

        self.class_kw_matrix = np.zeros((len(fnfc_classes), len(self.kw_vocab)))
        for i, cls in enumerate(fnfc_classes):
            kws = ISO_25010_KNOWLEDGE_BASE.get(cls, {}).get("keywords", [])
            for kw in kws:
                kw_l = kw.lower().strip()
                if kw_l in vocab_map:
                    idx = vocab_map[kw_l]
                    self.class_kw_matrix[i, idx] = 1.0

    def extract_iso_similarity_features(self, texts):
        text_matrix = self.tfidf.transform(texts)
        return cosine_similarity(text_matrix, self.profile_matrix) # (N, 14)

    def extract_keyword_density_features(self, texts):
        kw_counts = self.count_vec.transform(texts) # (N, V)
        density = kw_counts.dot(self.class_kw_matrix.T) # (N, 14)
        return density.toarray() if sp.issparse(density) else density

    def transform_all_features(self, texts):
        sim_feats = self.extract_iso_similarity_features(texts)
        kw_feats = self.extract_keyword_density_features(texts)
        return np.hstack([sim_feats, kw_feats]) # (N, 28)

