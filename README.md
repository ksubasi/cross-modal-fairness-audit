# Where Does the Asymmetry Enter?

Code and results for the paper **"Where Does the Asymmetry Enter? Staging a Fairness Audit Across a Fused Multimodal Moderation Pipeline"** by Kardelen Subasi and Jarosław A. Chudziak (Warsaw University of Technology), accepted at **PACLIC 40** (Taipei, December 2026).

The audit tests one subgroup axis, source language (Vietnamese, English, undetected), at three stages of MTikGuard, a published multimodal TikTok moderation classifier: the logged decisions, the pre-fusion branch representations, and the text the classifier is allowed to read after its token cap. Every stage is held to the same significance standard.

## Repository layout

| Path | Contents |
|---|---|
| `code/preprocess.py` | Builds the text CSV from the raw TikHarm videos (Whisper transcript, EasyOCR text, detected language) |
| `code/train.py` | Fine-tunes the MTikGuard architecture (`code/model.py`) with the published hyperparameters |
| `code/audit.py` | Decision-stage audit: per-subgroup FPR/FNR and the leave-one-modality-out comparison |
| `code/probe_subgroup.py` | Representation-stage probe with per-branch permutation nulls |
| `code/tokenization_audit.py` | Input-stage audit of the 64-word-piece text cap |
| `code/plot_truncation.py` | Truncation figure |
| `notebooks/mtikguard_pipeline.ipynb` | Runs preprocessing, training and the audit on Kaggle |
| `outputs/` | Audit results, per-item predictions and the pre-fusion features used by the probe |

## Reproducing the results

The released outputs are enough to rerun the representation-stage probe:

```bash
pip install -r code/requirements.txt
python code/probe_subgroup.py --features-dir outputs
```

The full pipeline needs a GPU and the raw videos from the public Kaggle dataset [`anhoangvo/tikharm-dataset`](https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset):

```bash
cd code
python preprocess.py --data-dir ../data/raw/Dataset
python train.py
python audit.py --checkpoint ../outputs/checkpoints/final
python tokenization_audit.py --csv ../data/processed/tikharm_vi_en_complete.csv
python probe_subgroup.py --features-dir ../outputs
```

The notebook runs the same steps on Kaggle.

## Differences from the original system

MTikGuard's preprocessed training table was never released, so `preprocess.py` rebuilds it from the public videos following the described extraction. Transcription uses Whisper-small, and language is detected with `langdetect`. MTikGuard's own pipeline computes each clip's language but does not store it; here it is kept as the subgroup column.

## Citation

```bibtex
@inproceedings{subasi2026asymmetry,
  title     = {Where Does the Asymmetry Enter? Staging a Fairness Audit Across a Fused Multimodal Moderation Pipeline},
  author    = {Subasi, Kardelen and Chudziak, Jaros{\l}aw A.},
  booktitle = {Proceedings of the 40th Pacific Asia Conference on Language, Information and Computation (PACLIC 40)},
  year      = {2026}
}
```
