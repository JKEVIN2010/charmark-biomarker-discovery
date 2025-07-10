# CharMark: Unmasking Cognitive Decline in Everyday Speech

Imagine a future where your casual conversation—describing a picture, chatting with a friend—could reveal the earliest whispers of cognitive change.  
**CharMark** brings that future a step closer. It’s a lightweight Python toolkit that turns plain text transcripts into interpretable “fingerprints” of your speech, helping researchers and clinicians spot subtle signs of dementia before they become unmissable.

---

## Why CharMark?

Traditional speech analyses dive into audio waves or complex neural nets—powerful, but often opaque. CharMark asks a different question:  
> **“At the level of single characters and pauses, how does language flow?”**

By building a first-order Markov chain from your transcript, then computing its steady-state probabilities, CharMark captures micro-patterns—like hesitation in pauses or recurring letter sequences—that can serve as early digital biomarkers of cognitive decline.

---

## Key Features

- **Clean & Simple**: Load your CSV of transcripts, call a few methods, and get back matrices of interpretable features.  
- **Interpretable**: Character-level probabilities, Kolmogorov-Smirnov tests, PCA plots, and network visualizations lay bare why two groups differ.  
- **End-to-End Pipeline**: From raw text cleaning to unsupervised clustering and Lasso-based validation, everything’s in one place.  
- **Lightweight**: No massive datasets or GPUs required—just Python and your transcripts.  

---

## Quickstart

```bash
git clone https://github.com/yourname/charmark-biomarker-discovery.git
cd charmark-biomarker-discovery
pip install -r requirements.txt
```
---

## 📝 Example Notebook

Want to see CharMark in action?  
Check out [`example.ipynb`](example.ipynb) for a step-by-step demo using a toy dataset.  

This notebook walks you through:  
1. Creating a sample CSV  
2. Extracting steady-state features  
3. Running PCA, KS tests, and Lasso validation  
4. Visualizing the character transition network

---

## Cite Us

If you use **CharMark** in your research, please cite:

> Mekulu K, Aqlan F, Yang H (2025). *CharMark: Character-Level Markov Modeling to Detect Linguistic Signs of Dementia.* Preprint.  
> DOI: [10.21203/rs.3.rs-6391300/v1](https://doi.org/10.21203/rs.3.rs-6391300/v1)

BibTeX:
```bibtex
@misc{Mekulu2025CharMark,
  author = {Kevin Mekulu and Faisal Aqlan and Hui Yang},
  title = {CharMark: Character-Level Markov Modeling to Detect Linguistic Signs of Dementia},
  year = {2025},
  doi = {10.21203/rs.3.rs-6391300/v1},
  url = {https://doi.org/10.21203/rs.3.rs-6391300/v1},
  note = {Preprint on Research Square}
}





