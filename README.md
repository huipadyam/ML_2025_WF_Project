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

(filename, execution instructions, and code workflow summary — handled by Hyeji)

### **Gradient Boosting Experiment Scenario**

(filename, execution instructions, and code workflow summary — handled by Hyewon)

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
