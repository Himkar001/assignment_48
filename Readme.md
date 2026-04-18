# 🏥 Medical Imaging Classification using Transfer Learning

**Week 08 · Friday Assignment**
---

# 📌 Project Overview

This project focuses on building a **chest X-ray classification system** using **transfer learning** under real-world clinical constraints:

* Small dataset (~520 samples)
* Class imbalance
* Need for **explainability**
* High cost of **false negatives**

We implement and compare:

* Feature Extraction
* Fine-Tuning
* Training from Scratch

And design a **clinical triage system** based on model confidence.

---

# 📂 Dataset

* `medical_imaging_meta.csv`
* Contains:

  * Image paths
  * Labels (5 conditions)
  * Metadata (hospital, image quality)
  * 30 unlabeled samples

---

# 🟢 EASY PART

---

## 🔹 Sub-step 1: Dataset Analysis

### Code

```python
df = pd.read_csv('/content/medical_imaging_meta.csv')
df.head()
```

```python
df['label'].value_counts()
```

```python
pd.crosstab(df['hospital'], df['label'])
```

```python
pd.crosstab(df['image_quality'], df['label'])
```

---

### Results

* Dataset contains ~520 samples across 5 classes
* Strong class imbalance observed
* Minority classes identified
* Site bias present (some hospitals dominate certain labels)
* Image quality varies across classes

---

### Interpretation

* Accuracy alone is misleading
* Minority classes risk **low recall**
* Dataset bias can affect generalization
* Must prioritize **clinical safety over accuracy**

---

## 🔹 Sub-step 2: Feature Extraction

### Code

```python
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 5)
```

```python
train_model(model, train_loader)
evaluate_model(model, val_loader)
```

---

### Results

* Fast training
* Good performance on majority classes
* Poor recall on minority classes

---

### Interpretation

* Model biased toward dominant classes
* High risk of **false negatives**
* Not safe for clinical deployment

---

# 🟡 MEDIUM PART

---

## 🔹 Sub-step 3: Fine-Tuning

### Code

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

```python
optimizer = Adam(model.parameters(), lr=1e-4)
```

---

### Results

* Improved recall on minority classes
* Better overall balance

---

### Interpretation

* Fine-tuning adapts model to medical domain
* Safer than feature extraction
* Slight overfitting risk

---

## 🔹 Sub-step 4: Explainability (Grad-CAM)

### Code

```python
cam = generate_gradcam(model, image, target_class)
show_gradcam(image, cam)
```

---

### Results

* Correct predictions → focus on lung regions
* Incorrect predictions → focus on noise / edges

---

### Interpretation

* Model is partially learning meaningful features
* Failures indicate need for better training/data

---

### Radiologist Explanation

> The model highlights lung regions when correct.
> In incorrect cases, it focuses on irrelevant areas, indicating limitations.

---

## 🔹 Sub-step 5: ME1 Prep + Inference

### Concept: Transfer Learning

Transfer learning reuses pretrained models to improve performance on small datasets by leveraging learned features.

---

### Interview Questions

**Q1:** Feature extraction vs fine-tuning?
**A:** Feature extraction freezes weights; fine-tuning updates them.

**Q2:** Why transfer learning in medical imaging?
**A:** Small datasets benefit from pretrained representations.

---

### Prediction Code

```python
probs = F.softmax(outputs, dim=1)
conf, preds = torch.max(probs, 1)
```

---

### Results

* Predictions generated for 30 unlabeled images
* Confidence scores assigned

---

### Interpretation

* High confidence → reliable
* Low confidence → requires review

---

# 🔴 HARD PART

---

## 🔹 Sub-step 6: Strategy Comparison

### Approaches Compared

1. Feature Extraction
2. Fine-Tuning
3. Training from Scratch

---

### Results

| Method             | Performance | Stability | Minority Recall |
| ------------------ | ----------- | --------- | --------------- |
| Feature Extraction | Medium      | High      | Low             |
| Fine-Tuning        | High        | Medium    | High            |
| From Scratch       | Low         | Low       | Very Low        |

---

### Interpretation

* Training from scratch fails due to small data
* Fine-tuning performs best overall

---

### Conclusion

Fine-tuning is the **optimal strategy** for this dataset.

---

## 🔹 Sub-step 7: Triage System

### Code

```python
HIGH_CONF = 0.85
LOW_CONF = 0.60
```

```python
if conf >= 0.85:
    auto
elif conf >= 0.60:
    review
else:
    reject
```

---

### Results

* High confidence → auto-classified
* Medium → reviewed
* Low → rejected

---

### Interpretation

| Tier   | Action | Risk     |
| ------ | ------ | -------- |
| High   | Auto   | Low      |
| Medium | Review | Moderate |
| Low    | Reject | High     |

---

### Clinical Justification

* False negatives are dangerous
* Only high-confidence predictions are automated
* Human-in-the-loop ensures safety

---

### Final Insight

This system:

* Reduces risk
* Improves trust
* Supports radiologists

---

# ⚙️ How to Run

```bash
pip install torch torchvision pandas sklearn matplotlib
```

Run notebook sequentially:

1. Data loading
2. EDA
3. Model training
4. Evaluation
5. Triage

---

# 🧠 Key Takeaways

* Transfer learning is essential for small datasets
* Fine-tuning > feature extraction in medical tasks
* Accuracy is not enough → recall matters
* Explainability is critical in healthcare
* Confidence-based triage enables safe deployment

---

# ✅ Final Conclusion

This project demonstrates a **clinically-aware AI pipeline**:

* Data-driven decisions
* Model comparison
* Explainability
* Safe deployment strategy

The system prioritizes **patient safety over raw performance**, making it suitable for real-world medical applications.

---
