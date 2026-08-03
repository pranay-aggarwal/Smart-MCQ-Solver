# Smart MCQ Solver — DL & Generative AI Project

- **Name:** Pranay Aggarwal
- **Email ID:**  24f3004524@ds.study.iitm.ac.in
- **Roll No:** 24f3004524
- **Term:** T2-2026
- **Kaggle Competition:** Smart MCQ Solver Challenge

---

## 1. Project Overview
5-option (A–E) multiple-choice question answering task. Given a `prompt` and 5 candidate options, the model predicts the top-3 most likely correct options, scored by **MAP@3** (Mean Average Precision @ 3).


## 2. Models
| # | Type (rubric requirement) | Model | Notebook |
|---|---|---|---|
| 1 | Built from scratch | BiLSTM + Transformer Encoder | `model1_scratch_bilstm_transformer.ipynb` |
| 2 | Pretrained | `microsoft/deberta-v3-base` fine-tuned via `AutoModelForMultipleChoice` | `model2_deberta_finetune.ipynb` |
| 3 | Model of choice | TF-IDF (uni+bigrams) + Logistic Regression | `model3_tfidf_logreg.ipynb` |

All three models are tracked as separate runs in the same W&B project (`24f3004524-t22026`), logging common metrics.

## 3. Setup / Reproducibility
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
**Weights & Biases login**:
```bash
export WANDB_API_KEY=your_key_here
wandb login
```
**Running notebooks:** open any notebook in `notebooks/` and run top-to-bottom; each notebook is self-contained (data loading → EDA → preprocessing → model → training → W&B logging → Kaggle-format submission CSV).

## 4. Data
Competition data (`train.csv`, `test.csv`, `sample_submission.csv`) is loaded at runtime from the Kaggle competition input directory (`/kaggle/input/competitions/smart-mcq-solver-challenge/`) or via `kagglehub` and is **not** committed to this repository.

- `train.csv`: 2000 rows × [`id`, `prompt`, `A`, `B`, `C`, `D`, `E`, `answer`]
- `test.csv`: 500 rows × [`id`, `prompt`, `A`, `B`, `C`, `D`, `E`]

## 5. Evaluation Metric
**MAP@3** — top-3 ranked predictions per question; score = 1.0 (rank 1), 0.5 (rank 2), 1/3 (rank 3), 0 otherwise, averaged over all questions.
