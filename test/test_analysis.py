import Paper_Analysis


paper = """
Deep Convolutional Networks for Skin Lesion Classification

2023

Problem:
Classifying dermoscopic images of skin lesions into benign and
malignant categories.

Method:
A CNN-based pipeline trained end-to-end on dermoscopic images.

Dataset:
ISIC dataset (International Skin Imaging Collaboration).

Evaluation:
Accuracy, sensitivity, and AUC on a held-out test split.

Results:
The proposed CNN achieved 91.2% accuracy and 0.94 AUC,
outperforming a Random Forest baseline built on handcrafted
texture features.

Limitation:
Performance drops noticeably on darker Fitzpatrick skin tones
due to dataset imbalance.
"""


analysis = Paper_Analysis.analyze(
    paper_text=paper,
    paper_id="P01",
)

print("\n===== PAPER ANALYSIS =====\n")
print(analysis)