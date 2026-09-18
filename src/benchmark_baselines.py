"""
Controlled Apple-to-Apple Comparison Suite for FNFC Requirements Classification.
Precomputed fast features with optimized LightGBM parameters.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from iso_rag_retriever import ISORagRetriever

class PrecomputedRAGWrapper:
    def __init__(self, clf, use_rag=False):
        self.clf = clf
        self.use_rag = use_rag

    def fit(self, X_text_train, X_iso_train, y_train):
        if self.use_rag:
            X_train = sp.hstack([X_text_train, sp.csr_matrix(X_iso_train * 2.0)])
        else:
            X_train = X_text_train
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X_text_test, X_iso_test):
        if self.use_rag:
            X_test = sp.hstack([X_text_test, sp.csr_matrix(X_iso_test * 2.0)])
        else:
            X_test = X_text_test
        return self.clf.predict(X_test)


class PrecomputedEnsembleWrapper:
    def __init__(self, use_rag=False):
        self.use_rag = use_rag
        self.clf_lr = LogisticRegression(max_iter=300, class_weight='balanced', C=3.5, random_state=42)
        self.clf_lgb = lgb.LGBMClassifier(n_estimators=25, num_leaves=15, class_weight='balanced', learning_rate=0.1, random_state=42, verbosity=-1, n_jobs=-1)

    def fit(self, X_text_train, X_iso_train, y_train):
        if self.use_rag:
            fused_train = sp.hstack([X_text_train, sp.csr_matrix(X_iso_train * 2.5)])
            self.clf_lr.fit(fused_train, y_train)
            self.clf_lgb.fit(X_iso_train, y_train)
        else:
            self.clf_lr.fit(X_text_train, y_train)
            self.clf_lgb.fit(X_text_train, y_train)
        return self

    def predict(self, X_text_test, X_iso_test):
        if self.use_rag:
            fused_test = sp.hstack([X_text_test, sp.csr_matrix(X_iso_test * 2.5)])
            p_lr = self.clf_lr.predict_proba(fused_test)
            p_lgb = self.clf_lgb.predict_proba(X_iso_test)
            return np.argmax(0.55 * p_lr + 0.45 * p_lgb, axis=1)
        else:
            p_lr = self.clf_lr.predict_proba(X_text_test)
            p_lgb = self.clf_lgb.predict_proba(X_text_test)
            return np.argmax(0.50 * p_lr + 0.50 * p_lgb, axis=1)


def get_precomputed_models():
    models = {
        "MNB (Base)": PrecomputedRAGWrapper(MultinomialNB(alpha=0.1), use_rag=False),
        "MNB + ISO RAG": PrecomputedRAGWrapper(MultinomialNB(alpha=0.1), use_rag=True),

        "Linear SVM (Base)": PrecomputedRAGWrapper(LinearSVC(class_weight='balanced', C=1.0, random_state=42), use_rag=False),
        "Linear SVM + ISO RAG": PrecomputedRAGWrapper(LinearSVC(class_weight='balanced', C=1.0, random_state=42), use_rag=True),

        "Random Forest (Base)": PrecomputedRAGWrapper(RandomForestClassifier(n_estimators=25, class_weight='balanced', random_state=42, n_jobs=-1), use_rag=False),
        "Random Forest + ISO RAG": PrecomputedRAGWrapper(RandomForestClassifier(n_estimators=25, class_weight='balanced', random_state=42, n_jobs=-1), use_rag=True),

        "Logistic Regression (Base)": PrecomputedRAGWrapper(LogisticRegression(max_iter=300, class_weight='balanced', C=2.0, random_state=42), use_rag=False),
        "Logistic Regression + ISO RAG": PrecomputedRAGWrapper(LogisticRegression(max_iter=300, class_weight='balanced', C=2.0, random_state=42), use_rag=True),

        "LightGBM (Base)": PrecomputedRAGWrapper(lgb.LGBMClassifier(n_estimators=25, num_leaves=15, class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1), use_rag=False),
        "LightGBM + ISO RAG": PrecomputedRAGWrapper(lgb.LGBMClassifier(n_estimators=25, num_leaves=15, class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1), use_rag=True),

        "Extra Trees (Base)": PrecomputedRAGWrapper(ExtraTreesClassifier(n_estimators=25, class_weight='balanced', random_state=42, n_jobs=-1), use_rag=False),
        "Extra Trees + ISO RAG": PrecomputedRAGWrapper(ExtraTreesClassifier(n_estimators=25, class_weight='balanced', random_state=42, n_jobs=-1), use_rag=True),

        "Ensemble (Base)": PrecomputedEnsembleWrapper(use_rag=False),
        "Ensemble + ISO RAG (Ours)": PrecomputedEnsembleWrapper(use_rag=True)
    }
    return models
