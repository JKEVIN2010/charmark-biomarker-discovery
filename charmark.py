# charmark.py

import string
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import networkx as nx

__version__ = "0.1.0"


class CharMark:
    """
    CharMark: Character-level Markov modeling for linguistic biomarker discovery.
    """

    def __init__(self, filepath, alpha=0.01, transcript_col='transcript', label_col='label'):
        """
        Parameters
        ----------
        filepath : str
            Path to a CSV file with columns [transcript_col, label_col].
        alpha : float
            Laplace smoothing constant for transition probabilities.
        transcript_col : str
            Name of the column containing text transcripts.
        label_col : str
            Name of the column containing binary labels (0=control, 1=dementia).
        """
        self.filepath = filepath
        self.alpha = alpha
        self.transcript_col = transcript_col
        self.label_col = label_col

        # Define state space: a-z plus space
        self.chars = list(string.ascii_lowercase) + [' ']
        self.k = len(self.chars)
        self._idx = {c: i for i, c in enumerate(self.chars)}

        # placeholders
        self.X = None
        self.labels = None

    def load_data(self):
        """
        Load CSV, clean transcripts, and extract labels.
        
        Returns
        -------
        texts : list of str
        labels : np.ndarray
        """
        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=[self.transcript_col, self.label_col])
        texts = df[self.transcript_col].astype(str).apply(self._clean_text).tolist()
        self.labels = df[self.label_col].values
        return texts, self.labels

    def _clean_text(self, text):
        """
        Lowercase and remove any character not in a-z or space.
        """
        text = text.lower()
        return ''.join(c for c in text if c in self.chars)

    def _build_transition_matrix(self, seq):
        """
        Build 1st-order transition matrix with Laplace smoothing.
        
        Returns
        -------
        P : np.ndarray, shape (k, k)
            Markov transition probability matrix.
        """
        M = np.zeros((self.k, self.k), dtype=float)
        for a, b in zip(seq, seq[1:]):
            M[self._idx[a], self._idx[b]] += 1
        # Laplace smoothing
        M += self.alpha
        M /= M.sum(axis=1, keepdims=True)
        return M

    def _steady_state(self, P):
        """
        Compute steady-state distribution (left eigenvector) for P.
        
        Returns
        -------
        pi : np.ndarray, shape (k,)
        """
        w, v = eig(P.T)
        stat = np.real(v[:, np.isclose(w, 1)])
        pi = stat[:, 0]
        return pi / pi.sum()

    def fit_transform(self):
        """
        Compute steady-state feature matrix for all transcripts.
        
        Returns
        -------
        X : np.ndarray, shape (n_samples, k)
        labels : np.ndarray
        """
        texts, self.labels = self.load_data()
        features = []
        for seq in texts:
            P = self._build_transition_matrix(seq)
            pi = self._steady_state(P)
            features.append(pi)
        self.X = np.vstack(features)
        return self.X, self.labels

    def find_optimal_clusters(self, X=None, k_min=2, k_max=5):
        """
        Evaluate silhouette scores for k in [k_min, k_max].
        
        Returns
        -------
        scores : dict
            {k: silhouette_score}
        """
        from sklearn.metrics import silhouette_score
        X = X if X is not None else self.X
        scores = {}
        for k in range(k_min, k_max + 1):
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
            scores[k] = silhouette_score(X, labels)
        return scores

    def run_kmeans(self, X=None, n_clusters=2):
        """
        Perform k-means clustering.
        
        Returns
        -------
        labels : np.ndarray
        """
        X = X if X is not None else self.X
        km = KMeans(n_clusters=n_clusters, random_state=0)
        return km.fit_predict(X)

    def plot_pca(self, X=None, labels=None):
        """
        Plot 2D PCA projection of features, colored by labels.
        """
        X = X if X is not None else self.X
        labels = labels if labels is not None else self.labels
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.title('PCA of CharMark Features')
        plt.colorbar(label='Label')
        plt.show()

    def ks_test(self, X=None, labels=None):
        """
        Perform KS test on each character distribution.
        
        Returns
        -------
        results : dict
            {char: {'ks_stat': float, 'p_value': float}}
        """
        X = X if X is not None else self.X
        labels = labels if labels is not None else self.labels
        df = pd.DataFrame(X, columns=self.chars)
        results = {}
        for c in self.chars:
            d, p = ks_2samp(df.loc[labels == 1, c], df.loc[labels == 0, c])
            results[c] = {'ks_stat': d, 'p_value': p}
        return results

    def lasso_validation(self, X=None, labels=None, cv=5):
        """
        Train Lasso Logistic Regression with CV, plot ROC, return AUC.
        """
        X = X if X is not None else self.X
        labels = labels if labels is not None else self.labels
        clf = LogisticRegressionCV(penalty='l1', solver='saga', cv=cv,
                                   scoring='roc_auc', max_iter=5000).fit(X, labels)
        probs = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Lasso)'); plt.legend()
        plt.show()
        return roc_auc

    def plot_network(self, example_index=0, threshold=None):
        """
        Visualize a Markov transition network for one transcript.
        
        Parameters
        ----------
        example_index : int
            Index of transcript to visualize.
        threshold : float or None
            Minimum transition probability to include an edge.
        """
        texts, _ = self.load_data()
        seq = texts[example_index]
        P = self._build_transition_matrix(seq)
        G = nx.DiGraph()
        # default: top 5% edges
        thresh = threshold if threshold is not None else np.percentile(P, 95)
        for i, a in enumerate(self.chars):
            for j, b in enumerate(self.chars):
                if P[i, j] >= thresh:
                    G.add_edge(a, b, weight=P[i, j])
        pos = nx.spring_layout(G, seed=0, k=0.15)
        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, node_size=100)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=G.edges(),
            width=[d['weight'] * 5 for _, _, d in G.edges(data=True)]
        )
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title('Character Transition Network')
        plt.axis('off')
        plt.show()
