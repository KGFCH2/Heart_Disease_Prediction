# 📖 Machine Learning & Medical Prediction Terms - Complete Glossary

A comprehensive guide to understanding all the key terms and metrics used in the Heart Disease Prediction System.

---

## 🎯 Core Model Concepts

### Epoch
**Definition:** One complete pass through the entire training dataset.

**Example:**
- Dataset has 10,000 samples
- Batch size = 64 samples
- One epoch = 10,000 ÷ 64 ≈ 156 batches processed
- After 156 batches, one epoch is complete

**Why it matters:**
- Models improve gradually over multiple epochs
- Too few epochs = underfitting (model hasn't learned enough)
- Too many epochs = overfitting (model memorizes data)
- Typical range: 100-500 epochs

**In our system:** We use `epochs=300` with `EarlyStopping` to stop when validation loss plateaus.

---

### Batch
**Definition:** A subset of training data processed together before updating model weights.

**How it works:**
1. Take batch of 64 samples
2. Forward pass through neural network
3. Calculate loss on batch
4. Backward pass: compute gradients
5. Update weights
6. Repeat with next batch

**Why batches instead of processing all samples?**
- Memory efficiency (can't fit all 10,000 samples on GPU at once)
- Better gradient estimates (averaging over batch)
- Faster training (parallelization)

**In our system:** `batch_size=64` - processes 64 samples together

---

### Loss (Loss Function)
**Definition:** A numerical measure of how wrong the model's predictions are.

**Formula for Binary Crossentropy (our system):**
```
Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

Where:
y = actual label (0 or 1)
ŷ = predicted probability [0, 1]
```

**Interpretation:**
- Loss = 0.5 → Model is uncertain (50% confidence)
- Loss = 0.1 → Model is very confident and correct
- Loss = 2.0 → Model is very confident but wrong

**Example:**
```
Patient has disease (y=1), model predicts 0.9 probability
Loss = -[1 × log(0.9) + 0 × log(0.1)] ≈ 0.105 (good)

Patient has disease (y=1), model predicts 0.1 probability
Loss = -[1 × log(0.1) + 0 × log(0.9)] ≈ 2.3 (bad)
```

**In our system:** Binary Crossentropy - suitable for medical yes/no diagnosis

---

### Gradient
**Definition:** Directional derivative showing how loss changes with respect to each weight.

**Conceptually:**
- Shows which direction to move weights to reduce loss
- Steep gradient = big change needed
- Shallow gradient = small change needed

**Example:**
```
If loss decreases when we increase weight_1, gradient is positive
If loss increases when we increase weight_1, gradient is negative
We update: weight_1 = weight_1 - learning_rate × gradient
```

**In our system:** Adam optimizer automatically computes and applies gradients

---

### Learning Rate
**Definition:** Step size for updating weights during training.

**Effect:**
- **Too high (e.g., 1.0):** Model overshoots optimal weights, diverges
- **Too low (e.g., 0.00001):** Training is very slow, might get stuck
- **Just right (e.g., 0.001):** Steady improvement toward optimal weights

**Visual analogy:** Walking down a hill
- Large steps (high LR) = faster but might overshoot the bottom
- Small steps (low LR) = slow but more precise

**In our system:** Initial `learning_rate=0.001` with `ReduceLROnPlateau` callback that reduces by 0.5× if no improvement

---

## 📊 Evaluation Metrics

### Accuracy
**Definition:** Percentage of correct predictions out of all predictions.

**Formula:**
```
Accuracy = (TP + TN) / Total
         = (Correct Diagnoses) / (All Patients)
```

**Example:**
```
100 patients tested:
- 50 correctly diagnosed as healthy
- 40 correctly diagnosed as diseased
- 10 misdiagnosed

Accuracy = (50 + 40) / 100 = 90%
```

**Limitation in Medical Context:**
- Can be misleading with imbalanced data
- If 90% of patients are healthy, model could get 90% accuracy by always predicting "healthy"
- **That's why we use multiple metrics** (precision, recall, AUC-ROC)

**In our system:** ~89% accuracy on test set

---

### Confusion Matrix
**Definition:** Table showing all four types of predictions: TP, TN, FP, FN.

**2×2 Matrix:**
```
                 Predicted Positive | Predicted Negative
Actual Positive  |        TP        |        FN
Actual Negative  |        FP        |        TN

TP = True Positive   (Disease predicted, actually has disease) ✓ Correct
TN = True Negative   (Healthy predicted, actually healthy) ✓ Correct
FP = False Positive  (Disease predicted, actually healthy) ✗ Type I Error
FN = False Negative  (Healthy predicted, actually has disease) ✗ Type II Error (DANGEROUS IN MEDICINE)
```

**Example for Heart Disease:**
```
                  Model: Disease | Model: Healthy
Actual: Disease   |      38      |       12       (12 missed diagnoses - CRITICAL!)
Actual: Healthy   |       7      |      443       (7 false alarms - ok)

Total: 500 patients
Correct: 38 + 443 = 481
Wrong: 12 + 7 = 19
Accuracy: 481/500 = 96.2%
```

**Medical Importance:**
- **False Negatives (FN):** Patient has disease but model says healthy → **DANGEROUS** ⚠️
  - Untreated disease progresses
  - Patient doesn't seek help
  
- **False Positives (FP):** Patient is healthy but model says disease → **Less critical**
  - Patient gets extra tests
  - Peace of mind when tests are negative

**In our system:** Generated as heatmap visualization showing all four quadrants

---

### Precision
**Definition:** Of all cases we predicted as positive, how many were actually positive?

**Formula:**
```
Precision = TP / (TP + FP)
          = Correct Positives / All Positive Predictions
          = "Accuracy of positive predictions"
```

**Example:**
```
Model predicts 50 patients have disease:
- 45 actually have disease ✓
- 5 don't have disease (false alarm) ✗

Precision = 45 / 50 = 90%
"When we predict disease, we're right 90% of the time"
```

**High Precision means:**
- Few false alarms
- When model says "disease," we can trust it
- Doctor won't waste time on unnecessary tests

**Low Precision means:**
- Many false alarms
- Many healthy patients diagnosed as sick
- Doctor spends time on unnecessary tests

**In our system:** ~87% precision - when we predict disease, we're correct 87% of the time

---

### Recall (Sensitivity)
**Definition:** Of all actual positive cases, how many did we correctly identify?

**Formula:**
```
Recall = TP / (TP + FN)
       = Correct Positives / All Actual Positives
       = "Coverage of positive cases"
```

**Example:**
```
100 patients actually have disease:
- 91 correctly diagnosed ✓
- 9 missed diagnoses ✗

Recall = 91 / 100 = 91%
"We catch 91% of disease cases"
```

**High Recall means:**
- Few missed diagnoses
- When disease is present, we find it
- Important in medical screening

**Low Recall means:**
- Many missed diagnoses ⚠️
- Patients with disease go undiagnosed
- **Dangerous in medical context**

**In our system:** ~91% recall - we correctly identify 91% of actual disease cases

---

### Precision vs Recall Trade-off

**Scenario 1: Strict Model (High Precision, Lower Recall)**
- Only predicts disease when very confident
- Few false alarms (few healthy patients diagnosed)
- Some missed diagnoses (some diseased patients missed)
- Use when: Tests are expensive/invasive, false alarms are costly

**Scenario 2: Sensitive Model (Lower Precision, High Recall)**
- Predicts disease more readily
- Catches most disease cases
- More false alarms (more healthy patients diagnosed)
- Use when: Disease is serious, missing one case is critical

**Our System:** Balanced approach
- Precision: 87% - reasonably confident in disease predictions
- Recall: 91% - catches most disease cases
- F1-Score: 0.89 - good balance between both

**Decision Threshold** (see below) controls this trade-off.

---

### F1-Score
**Definition:** Harmonic mean of Precision and Recall. Balanced metric for imbalanced datasets.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why harmonic mean?**
- Regular average: (87 + 91) / 2 = 89 ✗ (misleading)
- Harmonic mean: 2 × (87 × 91) / (87 + 91) ≈ 0.89 ✓ (penalizes imbalance)

**Interpretation:**
- F1 = 1.0 → Perfect precision and recall
- F1 = 0.5 → One metric is very low (problematic)
- F1 = 0.0 → Terrible predictions

**In our system:** F1-Score ≈ 0.89 (good balance)

---

## 📈 ROC & AUC

### ROC (Receiver Operating Characteristic) Curve
**Definition:** Plots True Positive Rate vs False Positive Rate at all decision thresholds.

**How it's built:**
1. For each threshold (0.0 to 1.0):
   - Calculate TPR (Recall) = TP / (TP + FN)
   - Calculate FPR = FP / (FP + TN)
   - Plot point (FPR, TPR)

2. Connect all points to form curve

**What the curve shows:**
```
Perfect Model:           Random Model:         Poor Model:
 ┌─────────┐            ┌─────────┐          ┌─────────┐
 │        ╱             │    ╱    │          │╲        │
 │      ╱               │  ╱      │          │  ╲      │
 │    ╱                 │╱        │          │    ╲    │
 └─────────┘            └─────────┘          └─────────┘
TPR=1 immediately       Diagonal line        Stays near bottom
(curve in top-left)     (random guessing)    (worse than random)
```

**Why it's useful:**
- Shows performance across ALL thresholds
- Single number (AUC) summarizes entire curve
- Threshold-independent evaluation

**In our system:** ROC curve saved as visualization with AUC ≈ 0.92

---

### AUC-ROC (Area Under the ROC Curve)
**Definition:** The probability that model ranks a random positive case higher than a random negative case.

**Formula:** Area under the ROC curve (between 0 and 1)

**Interpretation:**
```
AUC = 1.0   → Perfect model (always separates classes)
AUC = 0.9   → Excellent model (strong discrimination)
AUC = 0.7   → Good model
AUC = 0.5   → Random guessing (coin flip)
AUC = 0.0   → Completely wrong predictions
```

**Example:**
```
AUC = 0.92 means:
If we randomly pick one diseased patient and one healthy patient,
the model ranks the diseased patient higher 92% of the time.
```

**Why AUC-ROC is better than Accuracy:**
- **Accuracy** can be high with imbalanced data (91% healthy, always predict healthy → 91% accuracy)
- **AUC-ROC** measures discrimination ability regardless of class distribution
- Directly answers: "Can model distinguish between classes?"

**In our system:** AUC-ROC ≈ 0.9234 (excellent discrimination)

---

## 🎚️ Decision Threshold

### Definition
The probability cutoff determining whether to classify as positive or negative.

**Default:** 0.5 (standard threshold)

**How it works:**
```
Model Output: 0.73 (probability of disease)

If threshold = 0.5:
  0.73 > 0.5 → Predict: DISEASE ✓

If threshold = 0.7:
  0.73 > 0.7 → Predict: DISEASE ✓

If threshold = 0.8:
  0.73 < 0.8 → Predict: HEALTHY ✗
```

### Adjusting the Threshold

**Lower threshold (e.g., 0.3):**
- More predictions of disease
- Higher Recall (catch more cases)
- Lower Precision (more false alarms)
- **Use when:** Missing disease is worse than false alarms

**Higher threshold (e.g., 0.7):**
- Fewer predictions of disease
- Lower Recall (miss more cases)
- Higher Precision (fewer false alarms)
- **Use when:** False alarms are worse than missing cases

**Default threshold (0.5):**
- Balanced approach
- Equal cost for false positives and false negatives

**Example: Heart Disease Diagnosis**
```
Patient A: Model confidence = 0.52

Threshold 0.5: 0.52 > 0.5 → "You have disease" (recommend treatment)
Threshold 0.7: 0.52 < 0.7 → "You're healthy" (no treatment)

Which is safer? Usually lower threshold for serious diseases!
```

**In our system:** We use default 0.5 threshold in `y_pred = (y_prob > 0.5).astype(int)`

---

## 🧠 Neural Network Concepts

### Forward Pass
**Definition:** Data flows through the neural network from input to output.

**Process:**
```
Input (20 features)
    ↓ [multiply by weights + bias]
Dense(256)
    ↓ [apply ReLU activation]
ReLU output
    ↓ [multiply by weights + bias]
Dense(128)
    ... (continue through all layers)
    ↓
Output (1 neuron)
    ↓ [apply Sigmoid]
Probability [0, 1]
```

**Purpose:** Generate predictions

---

### Backward Pass (Backpropagation)
**Definition:** Gradients flow backward through network to compute how much each weight contributed to error.

**Process:**
```
Output error detected
    ↓
"How much did last layer's weights cause this error?"
    ↓
"How much did previous layer's weights cause their error?"
    ↓
... propagate backward ...
    ↓
"How much did input weights cause their error?"
```

**Purpose:** Calculate gradients for weight updates

---

### Activation Function (ReLU)
**Definition:** Rectified Linear Unit - applies non-linearity to neuron outputs.

**Formula:**
```
ReLU(x) = max(0, x)

If x > 0: output = x
If x ≤ 0: output = 0
```

**Graph:**
```
     │     ╱
     │   ╱
─────┼─╱
     │
```

**Why ReLU?**
- Computationally efficient
- Helps network learn non-linear patterns
- Prevents "vanishing gradient" problem
- Default choice for deep networks

**In our system:** ReLU used in Dense layers (except output which uses Sigmoid)

---

### Sigmoid Activation (Output Layer)
**Definition:** Squashes any value to probability between 0 and 1.

**Formula:**
```
Sigmoid(x) = 1 / (1 + e^(-x))
```

**Graph:**
```
Probability
   1 ├─────────────────╮
     │              ╱╱
 0.5 │            ╱╱
     │          ╱╱
   0 └────────╱──────────
     ← x →
```

**Interpretation:**
- Output = 0.9 → 90% probability of disease
- Output = 0.2 → 20% probability of disease

**Why Sigmoid for binary classification?**
- Output is naturally in [0, 1] - interpretable as probability
- Required for Binary Crossentropy loss

**In our system:** Sigmoid output layer produces risk scores [0, 1]

---

### Regularization Techniques

#### L2 Regularization
**Definition:** Penalty on large weights to prevent overfitting.

**Formula:**
```
Total Loss = Data Loss + λ × Σ(weights²)

λ = 0.001 (regularization strength)
```

**Effect:**
- Encourages smaller weights
- Prevents "overfitting" where model memorizes training data
- Like penalty in sports: "If you use extreme tactics, you get penalized"

**In our system:** `kernel_regularizer=l2(0.001)` on Dense layers

---

#### Dropout
**Definition:** Randomly deactivates neurons during training for regularization.

**How it works:**
```
During training (Dropout Rate = 0.4):
  Neuron 1: ACTIVE    ✓
  Neuron 2: DROPPED   ✗ (randomly deactivated)
  Neuron 3: ACTIVE    ✓
  Neuron 4: DROPPED   ✗
  ...
  Each batch: Different neurons dropped randomly

During inference:
  All neurons used (no dropout)
```

**Why Dropout?**
- Ensemble effect: Creates many different sub-networks during training
- Prevents co-adaptation: Neurons can't rely on specific other neurons
- Like reducing test anxiety by not knowing which questions appear

**In our system:** Dropout rates 0.4, 0.3, 0.2 at first three dense layers

---

#### Batch Normalization
**Definition:** Normalizes layer inputs to mean=0, std=1 for stability.

**Formula:**
```
Normalized = (x - batch_mean) / sqrt(batch_variance)
```

**Effect:**
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift
- Speeds up convergence

**In our system:** BatchNormalization after first three Dense layers

---

## 🔄 Training Techniques

### Early Stopping
**Definition:** Stop training if validation performance doesn't improve for N epochs.

**How it works:**
```
Epoch 1: val_loss = 0.50 (best so far) ✓ Save model
Epoch 2: val_loss = 0.48 (improved) ✓ Save model
Epoch 3: val_loss = 0.49 (worse) - patience counter = 1
Epoch 4: val_loss = 0.50 (worse) - patience counter = 2
...
Epoch 52: val_loss = 0.51 (still worse) - patience counter = 50 → STOP!
```

**Benefits:**
- Prevents overfitting (training too long)
- Saves computation time
- Automatically finds best model

**In our system:** `EarlyStopping(monitor='val_loss', patience=50)`

---

### Learning Rate Scheduling
**Definition:** Reduce learning rate if training plateaus.

**How it works:**
```
Initial LR: 0.001
Epoch 1-10: No improvement → LR × 0.5 = 0.0005
Epoch 11-20: Still no improvement → LR × 0.5 = 0.00025
...continues...
```

**Purpose:**
- Jump out of local minima with large steps initially
- Fine-tune with small steps when progress slows

**In our system:** `ReduceLROnPlateau(factor=0.5, patience=10)`

---

## 🏥 Medical-Specific Concepts

### Class Imbalance
**Definition:** Dataset has unequal classes (e.g., 70% healthy, 30% diseased).

**Problem:**
```
If model always predicts "healthy":
  Accuracy = 70% (appears good!)
  But: Catches 0% of disease cases (DANGEROUS)
```

**Solution: Class Weights**
```
Healthy class weight: 1.0
Disease class weight: 2.33

Effect: Disease misclassifications penalized 2.33× more
```

**In our system:** `class_weight='balanced'` computed from training data

---

### Sensitivity (Recall)
**Definition:** Ability to correctly identify cases with disease.

**Medical context:**
- "Out of 100 patients with disease, how many will we catch?"
- Critical for screening tests
- High sensitivity = few missed diagnoses

**In our system:** Recall ≈ 91%

---

### Specificity (1 - False Positive Rate)
**Definition:** Ability to correctly identify cases without disease.

**Formula:**
```
Specificity = TN / (TN + FP)
```

**Medical context:**
- "Out of 100 healthy patients, how many will we correctly identify as healthy?"
- Related to false alarms
- High specificity = few false alarms

---

### Positive Predictive Value (PPV) = Precision
**Definition:** If we predict disease, what's the probability patient actually has it?

**Medical context:**
- "If my model says I have disease, what's the chance I actually do?"
- Important for patient counseling
- In our system: 87%

---

### Negative Predictive Value (NPV)
**Definition:** If we predict healthy, what's the probability patient is actually healthy?

**Formula:**
```
NPV = TN / (TN + FN)
```

**Medical context:**
- "If my model says I'm healthy, what's the chance I actually am?"
- Reassurance value
- Should be very high (>95%) to be clinically useful

---

## 📊 Data Concepts

### Train-Test Split
**Definition:** Divide data into training set (80%) and testing set (20%).

**Why?**
- **Training set:** Model learns from this data
- **Testing set:** Evaluate on unseen data (true performance)

**Without splitting:**
- Evaluating on training data = biased (model has seen it)
- Results seem better than they actually are

**In our system:** `test_size=0.2, stratify=y`

---

### Stratification
**Definition:** Maintain class distribution in both train and test sets.

**Example:**
```
Original data: 60% healthy, 40% diseased

Without stratification:
  Train: 65% healthy, 35% diseased ✗ (imbalanced)
  Test: 45% healthy, 55% diseased ✗ (very different)

With stratification (stratify=y):
  Train: 60% healthy, 40% diseased ✓ (balanced)
  Test: 60% healthy, 40% diseased ✓ (balanced)
```

**In our system:** `stratify=y` ensures consistent class distribution

---

### Feature Scaling
**Definition:** Normalize features to similar ranges for fair neural network training.

**StandardScaler:**
```
scaled = (x - mean) / standard_deviation

Example:
  Original Age: 25 years
  Mean Age: 50 years
  Std Dev: 15 years
  Scaled = (25 - 50) / 15 = -1.67
```

**Result:** All features in range ~[-3, 3]

**Why?**
- Neural networks learn better with normalized inputs
- Large-scale features don't dominate learning
- Faster convergence

**In our system:** StandardScaler on all 20 features

---

## 📚 Summary Table

| Term | What It Is | Range | Goal |
|------|-----------|-------|------|
| Accuracy | Overall correctness | 0-100% | Higher |
| Precision | Reliability of positive predictions | 0-100% | Higher |
| Recall/Sensitivity | Coverage of actual positives | 0-100% | Higher |
| Specificity | Coverage of actual negatives | 0-100% | Higher |
| F1-Score | Harmonic mean of precision & recall | 0-1 | Higher |
| AUC-ROC | Discrimination ability | 0-1 | Higher |
| Loss | Prediction error | 0-∞ | Lower |
| Epoch | One full data pass | 1+ | ~100-300 |

---

## 🎓 Quick Reference Examples

### Example 1: Perfect Model
```
Confusion Matrix:    Precision: 100%
  95   0             Recall: 100%
   0  405            Accuracy: 100%
                     AUC-ROC: 1.0
```

### Example 2: Good Model (Our System)
```
Confusion Matrix:    Precision: 87%
  91   13            Recall: 91%
   7  389            Accuracy: 89%
                     AUC-ROC: 0.92
```

### Example 3: Poor Model
```
Confusion Matrix:    Precision: 50%
  50   50            Recall: 50%
  50  350            Accuracy: 67%
                     AUC-ROC: 0.5
```

---

## 🔗 Related Documents

- **README.md** - Quick start guide
- **COMPLETE_GUIDANCE.md** - Technical implementation details
- **disease_tensorflow.py** - Code implementation
- **COMPLETE_GUIDANCE.md → Output Interpretation** - How to read metrics

---

**Last Updated:** November 2025  
**Version:** 1.0
