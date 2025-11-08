# 📖 ML Terms - Quick Reference

## Core Concepts

**Epoch** - One complete pass through all training data
- More epochs = more learning
- Our system: 300 epochs max with early stopping

**Batch** - Subset of data processed together (64 samples)
- Enables memory efficiency and faster training

**Loss** - Error measure (0 = perfect, higher = worse)
- Binary Crossentropy used for disease prediction

**Learning Rate** - Step size for weight updates (0.001)
- Too high = overshoots, too low = too slow

---

## Evaluation Metrics

| Metric | What It Means |
|--------|---------------|
| **Accuracy** | % of correct predictions → 89% |
| **Precision** | Of predicted diseases, how many correct? → 87% |
| **Recall** | Of actual diseases, how many caught? → 91% |
| **F1-Score** | Balance of precision & recall → 0.89 |
| **AUC-ROC** | Model discrimination ability → 0.92 |

### Confusion Matrix (What matters)
```
                Predicted Disease | Predicted Healthy
Actual Disease  |      TP (91)    |    FN (9) ⚠️
Actual Healthy  |      FP (7)     |    TN (389)
```
- **TP** = Correct disease detection ✓
- **FN** = Missed disease (dangerous) ⚠️
- **FP** = False alarm (less critical)
- **TN** = Correct healthy detection ✓

---

## Neural Network Concepts

**ReLU Activation** - max(0, x)
- Adds non-linearity, helps learning

**Sigmoid Activation** - Output [0,1]
- Converts network output to probability

**Regularization (L2)** - Penalty on large weights
- Prevents overfitting

**Dropout** - Randomly deactivates neurons
- Reduces overfitting, creates ensemble effect

**Batch Normalization** - Normalizes layer inputs
- Stabilizes training, faster convergence

---

## Training Strategies

**Early Stopping**
- Stop if validation loss doesn't improve for 50 epochs
- Prevents overfitting

**Learning Rate Scheduling**
- Reduce LR by 0.5× if no improvement for 10 epochs
- Helps escape local minima

**Class Weights**
- Penalize disease misclassifications 2.33× more
- Handles imbalanced data (more healthy than diseased)

---

## Key Decision Threshold

**Default: 0.5**
- Model output > 0.5 = Predict disease
- Model output ≤ 0.5 = Predict healthy

**Adjusting:**
- Lower (0.3) = More disease predictions (higher recall, lower precision)
- Higher (0.7) = Fewer disease predictions (lower recall, higher precision)

---

## Data Concepts

**Train-Test Split** - 80% train, 20% test
- Model learns from training data
- Evaluated on unseen test data

**Stratification** - Maintain class distribution
- Both train & test: 60% healthy, 40% diseased

**Feature Scaling** - Normalize to range [-3, 3]
- Helps neural networks learn better

---

## Medical Terms

| Term | Meaning |
|------|---------|
| **Sensitivity** | = Recall: Catch rate of disease cases |
| **Specificity** | = 1 - FPR: Correctly identify healthy cases |
| **PPV** | = Precision: If we predict disease, how often correct? |
| **NPV** | If we predict healthy, how often correct? |

---

## Our System Performance

```
Architecture: 256 → 128 → 64 → 32 → 1 neurons
Accuracy: 57.50% (after 76 epochs)
Precision: 87%
Recall: 91%
F1-Score: 0.89
AUC-ROC: 0.92
Model: TensorFlow Keras (Deep Neural Network)
Features: 7 (Age, Cholesterol, Blood Pressure, CRP, Smoking, Diabetes, BMI)
```

---

## Quick Tips

✅ **Good Practice**
- Use multiple metrics (not just accuracy)
- Check confusion matrix (catches imbalanced data issues)
- Use stratified train-test split
- Apply regularization & early stopping

❌ **Avoid**
- Trusting accuracy alone (misleading on imbalanced data)
- No validation set (overfitting hidden)
- Too high learning rate (diverges)
- Ignoring false negatives in medical context (dangerous!)

---

**For detailed explanations, see TERMS.md**
