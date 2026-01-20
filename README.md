## Two-stage NPF classifier (strict gate)

- Stage 1: Logistic Regression gate predicts event vs nonevent (class2)
- Stage 2: RBF SVM predicts subtype (Ia/Ib/II/nonevent) with strict gating
- Outputs Kaggle submission with:
  - class4 prediction
  - p = P(event) from Stage 1

Dataset is not included because it was provided by a course.

### Run
pip install -r requirements.txt
python your_script.py --train data/train.csv --test data/test.csv --out submission.csv