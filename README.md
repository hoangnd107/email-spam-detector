## Email Spam Detector

Spam emails are a major challenge for email services and users. This project focuses on building a machine learning model that efficiently classifies emails into spam or non-spam categories. By analyzing the textual content of emails and applying NLP preprocessing, the model achieves high accuracy and robustness against various spam patterns.

![Pipeline of Spam/Ham Email Detector](https://user-images.githubusercontent.com/80247118/209618599-e6acd06d-0a4d-49a6-bf1e-0ae83b7da6d5.png)

---

## Dataset

The dataset used for training and testing the model consists of labeled emails categorized into `spam` and `ham`. Popular datasets like the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) or any publicly available email dataset can be utilized.

---

## Technologies Used

- **Python**, **Jupyter Notebook**, `pandas`, `matplotlib`, `seaborn`, `nltk`, `over_sampling`, `scikit-learn`.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/hoangnd107/email-spam-detector.git
   cd email-spam-detector
   ```
2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), change the data file extension to .txt and place it in the directory.

4. Run the `source.ipynb` file step-by-step.

---

## Implementation Steps
1. **Data Loading**
2. **Data Preprocessing**:
   - Clean text data by removing punctuations, and numbers.
   - Tokenize text into words and remove stopwords.
   - Apply stemming/lemmatization to reduce words to their root forms.
3. **Feature Engineering**:
   - Convert text to numerical representation using Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF).
4. **Model Training and Testing**:
    - Split the dataset into training and testing sets.
    - Train and test models using Naive Bayes, SVM.
    - Hyperparameter optimization using Grid Search CV.
5. **Evaluation**:
   - Plot confusion matrices for visual performance insights.
   - Measure accuracy, precision, recall, and F1-score.

6. **Model Saving**

---

## Results

The Naive Bayes model achieved an accuracy of 96%, while the SVM model reached 99%.
