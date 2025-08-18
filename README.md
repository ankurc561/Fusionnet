# Fusionnet

Predicting Financial Market using Filings and News Sentiment

## Environment setup

- Create a fresh Python 3.10+ environment (Conda recommended):
  - `conda create -n fusionnet python=3.10 -y`
  - `conda activate fusionnet`
- Install dependencies:
  - `pip install -r requirements.txt`
- Optional: register the kernel for Jupyter
  - `python -m ipykernel install --user --name fusionnet --display-name "Python (fusionnet)"`

## Required credentials and configuration

Some notebooks require external services. Set these before running:

- Google Cloud (BigQuery/Storage):
  - `GOOGLE_CLOUD_PROJECT`: your GCP project ID
  - `GOOGLE_APPLICATION_CREDENTIALS`: path to your service account JSON key
- Google Gemini API (for labeling):
  - `GEMINI_API_KEY`: API key for `google-generativeai`

Ensure the BigQuery dataset and Storage buckets referenced in notebooks exist, or update notebook constants accordingly.

## Data flow and notebook order

1. Filings acquisition and parsing

   - `Filings_Crawler.ipynb`: Retrieves S&P 500 tickers and fetches filings metadata/content (EDGAR). Outputs raw JSON/objects.
   - `Filings_Dataset_parsing.ipynb`: Parses crawler outputs into tabular datasets (per filing section), ready for labeling.

2. Gemini labeling

   - `Filings_labelling.ipynb`: Uses Gemini to label filing text chunks; writes results to BigQuery.
   - `News_Gemini_labeling.ipynb`: Uses Gemini to label news articles; writes results to BigQuery.

3. Model fine-tuning

   - `FinBERT_FineTune_Filings.ipynb`: Fine-tunes FinBERT on labeled filings.
   - `FinBERT_FineTune_News.ipynb`: Fine-tunes FinBERT on labeled news.
   - `DeBERTa_FineTune_Filings.ipynb`: Fine-tunes DeBERTa on labeled filings.
   - `DeBERTa_FineTune_News.ipynb`: Fine-tunes DeBERTa on labeled news.

4. Human-reviewed evaluation

   - `filings_human_eval.ipynb`: Evaluation for filings with human reviewed set.
   - `news_human_eval.ipynb`: Evaluation for news articles with human reviewed set.

5. Scoring with fine-tuned models

   - `filings_score.ipynb`: Scores filings using the fine-tuned FinBERT model.
   - `news_score.ipynb`: Scores news using the fine-tuned DeBERTa model.

6. Exploratory data analysis (EDA)

   - `Fusion_News_EDA.ipynb`: EDA of news labels/scores over time.
   - `Fusion_filings_EDA_initial.ipynb` and `Fusion_filings_EDA_current.ipynb`: EDA of filing labels/scores.

7. Modelling with weights

   - `Modelling_with_weights.ipynb`: Weighted aggregation of sentiment signals for filings with EDA.

8. Fusion and classification experiments (A1–A5)

   - `A1_randomWalk_splits.ipynb`: Random walk benchmark splits for robust evaluation.
   - `A2_statistical_significance.ipynb`: Statistical significance tests on model outputs.
   - `A3_Calibrated RF.ipynb`: Random Forest with calibration experiments.
   - `A4_Ablations.ipynb`: Ablation studies across features/models.
   - `A5_Backtest.ipynb`: Economic backtest using modeled signals.

## Notes on running

- GPU acceleration: Install a CUDA-enabled build of PyTorch if using GPU. Refer to PyTorch install selector to match your CUDA toolkit.
- Hugging Face models: The notebooks use `transformers` and `datasets`. Make sure to authenticate to Hugging Face if private models are used.
- BigQuery & Storage: Tables/datasets defined in notebooks (e.g., `edgar_sentiment.*`) must exist or will be created if the account is authorized. Update dataset/table names if needed.
- Reproducibility: Set random seeds in modeling notebooks as provided to replicate results.
- Large data: Some EDA notebooks expect precomputed scores stored in BigQuery/Storage; run the scoring notebooks first or adjust inputs.
