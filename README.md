# **Website Fingerprinting Classification — Lossless Team**

## **Team Members**

* **Kim Chaeeun (2130010)**
* **Kim Minsu (2130005)**
* **Park Yeonsu (2276131)**
* **Seo Hyeji (2171019)**
* **Lee Seoyoung (2276218)**
* **Choi Hyewon (2176394)**

---

## **1. Problem Definition**

### **Website Fingerprinting**

* **What**: Inferring visited websites from encrypted traffic packets
* **How**: Analyzing packet size, directionality, and timing patterns — effective even on Tor
* **Why**: Reveals the anonymity limitations of Tor

### **Project Goal**

Using machine learning to **simulate and analyze Website Fingerprinting attacks**,
and exploring the limitations and feasibility of models in both Closed-world and Open-world scenarios.

---

## **2. Experimental Overview**

This project consists of the following two main steps

1. **Closed-world**: Classification of 95 monitored websites
2. **Open-world**: Binary classification of monitored vs. unmonitored traffic

All experiments use **19 handcrafted features** (packet count, timing, directionality, initial phase, cumulative flow).

---

## **3. How to Run**

All notebooks can be executed directly in **Google Colab**.
Open each notebook file and run the cells sequentially.

### **Decision Tree Experiment Scenario**

(filename, execution instructions, and code workflow summary — handled by Minsu)

### **SVM Open-world Experiment Scenarios**

* `SVM_1to5.ipynb`
* `SVM_1to10.ipynb`
* `SVM_1to100.ipynb`

("1toN" represents the monitored : unmonitored ratio, reflecting real-world imbalance.)

1. **Load and preprocess data**
2. **Extract 19 features**
3. **Closed-world SVM experiment**

   * Coarse-to-fine grid search for C and gamma
4. **Open-world SVM experiment**

   * Coarse-to-fine grid search for C and gamma
   * Repeated under imbalance ratios 1:5 / 1:10 / 1:100
5. **Feature reduction experiment**

   * Compute feature importance using Random Forest
   * Retrain SVM using only top-k selected features

### **Random Forest Experiment Scenario**

* `RandomForest.ipynb`

1. **Load and preprocess data**

2. **Extract 19 features**
   
3. **Closed-world RandomForest experiment**
   * Hyperparameter Tuning - n_estimators, max_features, max_depth

4. **Feature Importance Analysis (Closed-world)**
   * Compute feature importance for the Closed-world experiment

5. **Open-world RandomForest experiment (Binary Classification)**
   * Set class_weight='balanced' to prevent bias toward the majority class (unmonitored) for the model training

6. **Open-world RandomForest experiment (Multi-class Classification)**
   * Data Preparation for Multi-class classification in Open-world
   * Set class_weight='balanced' to prevent bias toward the majority class (unmonitored) for the model training

7. **Feature Importance Analysis (Open-world)**
   * Compute feature importance for the Open-world experiment

### **Gradient Boosting Experiment Scenario**

* `Gradient_Boosting.ipynb`

* We selected Gradient Boosting as our primary model because it is a robust tree-based ensemble method capable of capturing nonlinear relationships within network traffic features. Unlike linear models or distance-based classifiers, Gradient Boosting incrementally corrects errors using a sequence of weak decision trees, making it well-suited for noisy, real-world packet patterns. Its strong generalization ability and interpretability (via feature importance and SHAP analysis) also make it an appropriate choice for our monitored-site detection task.

1. **Load and preprocess data**
   * Merge monitored + unmonitored traffic datasets
   * Extract 19 statistical traffic features
   * Apply StandardScaler
   * Stratified train/test split

2. **Closed-world Experiment (95-class classification)**
   * Model: GradientBoostingClassifier
   * Task: Predict which website among 95 monitored sites

- GridSearchCV (3-fold):
- n_estimators ∈ {100, 200}
- learning_rate ∈ {0.05, 0.1}
- max_depth ∈ {2, 3}
- subsample ∈ {0.8, 1.0}

   * Hyperparameter Search Space

| Hyperparameter    | Values Tested | Rationale |
|------------------|--------------|-----------|
| **n_estimators**  | 100, 200      | Controls model capacity; 100–300 is commonly effective in boosting literature. |
| **learning_rate** | 0.05, 0.1     | Balances underfitting vs. overfitting; these values provide stable convergence. |
| **max_depth**     | 2, 3          | Shallow trees help prevent overfitting and generalize better in boosting. |
| **subsample**     | 0.8, 1.0      | 0.8 adds helpful stochasticity to reduce variance; compared against full sampling. |



   * Metrics: Accuracy (baseline vs tuned)
   * Result: Gradient Boosting is not suitable for 95-class website fingerprinting
    Hyperparameters do not fix structural limitations

3. **Open-world Experiment (Binary: Monitored vs Unmonitored)**
   * Same feature preprocessing
   * Train Gradient Boosting to classify: 1 → Monitored, 0 → Unmonitored

   * Report: Precision / Recall / F1, ROC-AUC, Confusion matrix
   * Result: Near-perfect detection, False positives and false negatives extremely low

4. **Feature Interpretation (SHAP + Feature Importance)**
   * Identify key features the GB model relies on: Burst counts, Cumulative packet size (cum_abs_sum), Timing statistics
   * SHAP is executed only for open-world because the multi-class closed-world SHAP support is limited

   * Insight: Binary monitored/unmonitored separation is highly driven by burst + packet volume signals
             These features are easily separable → explains high open-world accuracy

---

## **4. Experimental Environment**

All experiments were conducted in the **Google Colab Free** environment.

* CPU: 2 vCPUs (AMD EPYC 7B12)
* RAM: ~13 GB
* GPU: None
* Python: 3.12
* Libraries: scikit-learn 1.6.1, numpy 2.0.2, pandas 2.2.2
* Execution time: ~300–600 seconds

---

## **5. Notes**

* All experiments are reproducible using a fixed random seed.
