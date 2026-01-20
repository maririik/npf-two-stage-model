## Two-stage NPF classifier (strict gate)

This project builds a two-stage machine learning model to detect **new particle formation (NPF)** events in atmospheric measurement data. NPF refers to the formation of new aerosol particles in the air, which can affect air quality and climate.

### Model
- **Stage 1 (Logistic Regression):** predicts **event vs nonevent** (class2) and outputs **p = P(event)**
- **Stage 2 (RBF SVM):** predicts **subtype** (Ia/Ib/II)  
  Used only when Stage 1 predicts an event (strict gating). Otherwise the final label stays **nonevent**.

The script outputs a Kaggle-style submission file with:
- `class4` prediction
- `p` = `P(event)` from Stage 1

Dataset is not included because it was provided by a course.

### Run
```bash
pip install -r requirements.txt
python src/train_predict.py --train data/train.csv --test data/test.csv --out submission.csv
makefile
Copy code
::contentReference[oaicite:0]{index=0}