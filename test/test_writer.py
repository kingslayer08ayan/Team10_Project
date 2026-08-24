import Paper_Analysis
import Comparison
import Writer


paper_1 = """
Deep Convolutional Networks for Skin Lesion Classification

2023

Problem:
Classifying dermoscopic images of skin lesions into benign and
malignant categories.

Method:
A CNN-based pipeline trained end-to-end on dermoscopic images.

Dataset:
ISIC dataset.

Evaluation:
Accuracy, sensitivity, and AUC.

Results:
The CNN achieved 91.2% accuracy and 0.94 AUC.

Limitation:
Performance drops on darker Fitzpatrick skin tones because of
dataset imbalance.
"""


paper_2 = """
Hybrid Texture-and-Deep-Feature Fusion for Melanoma Detection

2022

Problem:
Detecting melanoma from dermoscopic images under limited labeled data.

Method:
Combines Gabor filters, Local Binary Patterns and Histogram of
Oriented Gradients with a Random Forest classifier.

Dataset:
HAM10000.

Evaluation:
Precision, recall and F1-score.

Results:
The feature-fusion approach achieved an F1 score of 0.83,
compared with 0.86 for a CNN baseline.

Limitation:
The approach still performs slightly worse than the CNN baseline.
"""


print("===== PAPER ANALYSIS =====")

analysis_1 = Paper_Analysis.analyze(
    paper_text=paper_1,
    paper_id="P01",
)

analysis_2 = Paper_Analysis.analyze(
    paper_text=paper_2,
    paper_id="P02",
)

analyses = [
    analysis_1,
    analysis_2,
]


print("===== COMPARISON =====")

comparison = Comparison.compare(analyses)


print("===== LITERATURE REVIEW =====")

review = Writer.write(
    analyses=analyses,
    comparison=comparison,
)

print("\n")
print(review)