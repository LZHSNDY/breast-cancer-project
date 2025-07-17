# Breast Cancer Radiomics Project

This project explores the use of classical machine learning and radiomics techniques for breast cancer detection using medical imaging data (MRI, DICOM, etc.). It includes classification models, radiomics feature extraction, and data visualization, using datasets such as Duke MRI (NIfTI format) and CBIS-DDSM (DICOM).

---

##  Project Structure 

```
 notebooks/
│   ├── 01_read_nifti.ipynb         # Read & visualize NIfTI format images
│   ├── 02_read_dicom.ipynb         # Read & visualize DICOM format images
│   ├── 03_classification_models.ipynb  # Logistic, kNN, SVM, Tree, Naive Bayes
│   └── 04_radiomics_analysis.ipynb # Radiomics feature extraction and selection
 slides/
│   └── final_presentation.pdf      # Project summary & presentation
README.md                          # Project description
```

---

##  Datasets Used 

### 1. [Duke Breast Cancer MRI (NIfTI)](https://www.kaggle.com/datasets/madhava20217/duke-breast-cancer-mri-nifti-pre-and-post-1-only)
- Format: `.nii` / `.nii.gz`
- Used for radiomics analysis and visualization

### 2. [CBIS-DDSM Breast Cancer Dataset (DICOM/JPG)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- Format: `.dcm`, `.jpg`
- Used for training classical classification models

---

## 🖼️ Medical Image Visualizations

### 🧠 NIfTI Multi-view Slices

Below is an example showing axial, sagittal, and coronal views from a brain MRI NIfTI image:

![NIfTI Multi-directional Slices](images/nifti_multiview.png)

### 🧾 NIfTI Header Information

```
affine matrix:
[[  -1.    0.    0.   90. ]
 [   0.    1.    0. -126. ]
 [   0.    0.    1.  -72. ]
 [   0.    0.    0.    1. ]]
```
![NIfTI Header Screenshot](images/nifti_header.png)

### 📚 Mosaic View of Slices

A mosaic visualization gives a comprehensive overview of the entire volume:

![NIfTI Mosaic View](images/nifti_mosaic.png)

### 🔧 Visualization Tool Interface (e.g., BrainVoyager)

Example of how NIfTI scans appear in radiology visualization tools:

![Tool Interface Screenshot](images/nifti_tool_view.png)

> Image Sources: [ResearchGate](https://www.researchgate.net), [NeuralDataScience.io](https://neuraldatascience.io), [BrainVoyager](https://www.brainvoyager.com)

---

##  Models Implemented 

Implemented and compared the following classical machine learning models:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree & Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

Metrics used:
- Accuracy, Confusion Matrix, ROC Curve, AUC

---

##  Radiomics Feature Extraction 

Used `pyradiomics` to extract:
- GLSZM (Gray Level Size Zone Matrix)
- GLDM (Gray Level Dependence Matrix)
- GLRLM (Gray Level Run Length Matrix)
- First Order Statistics

Performed statistical significance filtering using `SelectKBest(f_classif)` from `sklearn`.

---

##  Presentation Highlights 

1. Dataset overview and NIfTI/DICOM format explanation
2. Model architecture and performance analysis
3. Radiomics pipeline and feature analysis
4. Lessons learned and future improvements

---

##  What I Learned 

- Learned how to handle medical image formats (NIfTI, DICOM)
- Understood classical machine learning models and performance metrics
- Gained experience in radiomics and statistical feature selection
- Practiced project organization and technical presentation

---

##  Instructor Questions 

1. **What are the different types of mammograms?**
   - Screening vs Diagnostic
   - 2D vs 3D (Tomosynthesis)
   - BI-RADS density categories (A–D)

2. **Which model performed best and why?**
3. **What are the most important radiomic features?**

---

##  Notebooks

All notebooks are available in the `notebooks/` directory with full code and explanations.

###  01_read_nifti.ipynb
```python
!pip install nibabel matplotlib

import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load('/path/to/image.nii')
data = img.get_fdata()

slice_idx = data.shape[2] // 2
plt.imshow(data[:, :, slice_idx], cmap='gray')
plt.title("Middle Slice of NIfTI Image")
plt.axis('off')
plt.show()
```

###  02_read_dicom.ipynb
```python
!pip install pydicom matplotlib

import pydicom
import os
import matplotlib.pyplot as plt

file_path = '/path/to/image.dcm'
ds = pydicom.dcmread(file_path)
plt.imshow(ds.pixel_array, cmap='gray')
plt.title(f"Patient: {ds.PatientID}")
plt.axis('off')
plt.show()

print("Pixel Spacing:", ds.PixelSpacing)
print("Modality:", ds.Modality)
```

###  03_classification_models.ipynb
```python
!pip install scikit-learn pandas matplotlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
  "Logistic": LogisticRegression(),
  "kNN": KNeighborsClassifier(),
  "Decision Tree": DecisionTreeClassifier(),
  "Random Forest": RandomForestClassifier(),
  "SVM": SVC(probability=True),
  "Naive Bayes": GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

###  04_radiomics_analysis.ipynb
```python
!pip install pyradiomics

from radiomics import featureextractor
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableFeatureClassByName("glszm")
extractor.enableFeatureClassByName("gldm")
extractor.enableFeatureClassByName("glrlm")
extractor.enableFeatureClassByName("firstorder")

result = extractor.execute(imagePath, maskPath)

features_df = pd.DataFrame([result])

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(features_df.drop(['Label'], axis=1), features_df['Label'])
print("Top selected features:", selector.get_support(indices=True))
```

---

*This project was completed as part of a medical imaging ML task focused on understanding and applying radiomics techniques for breast cancer classification.*
