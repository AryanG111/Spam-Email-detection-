The model achieves exceptional performance, validated through robust testing methods:

- **Test Accuracy:** `0.99`
- **5-Fold Cross-Validation Mean Accuracy:** `0.985`
- **Spam Detection Recall:** `0.99` (Catches 99% of all spam messages)
- **Spam Detection Precision:** `0.97` (97% of its spam predictions are correct)

These results indicate a highly reliable model that effectively balances catching spam and avoiding false positives.

## 🧰 Tech Stack

- **Programming Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` (Data Handling)
  - `scikit-learn` (Machine Learning Pipeline: `TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `cross_val_score`)
  - `nltk` (Text Preprocessing - Stopword Removal)
  - `re` (Regular Expressions for text cleaning)

## 📁 Project Structure
spam-sms-classifier/
├── spam_classifier.ipynb # Main Jupyter notebook with full analysis
├── requirements.txt # List of dependencies to reproduce the environment
└── README.md # Project overview and documentation


## 🚀 How It Works

The project follows a standard ML workflow:

1.  **Data Preprocessing:**
    - Text is lowercased and stripped of non-alphabet characters.
    - Common stopwords (e.g., "the", "and", "is") are removed to focus on meaningful tokens.
2.  **Feature Engineering:**
    - Text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which evaluates the importance of a word in a message relative to the entire dataset.
3.  **Model Training:**
    - A **Logistic Regression** classifier is trained on the processed features. This model was chosen for its efficiency and strong performance on linearly separable text data.
4.  **Evaluation:**
    - Performance is rigorously assessed using a **train-test split** and **5-fold cross-validation** to ensure generalizability and avoid overfitting.
    - A detailed **classification report** provides metrics for each class (spam/ham).

## ▶️ Getting Started

### Prerequisites

Ensure you have Python 3 and `pip` installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AryanG111/Spam-Email-detection-.git
    cd sms-spam-classifier
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK Stopwords:**
    Run the following command in your Python environment to download the necessary stopwords corpus:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

### Running the Project

Execute the Jupyter Notebook to see the entire process:
```bash
jupyter notebook spam_classifier.ipynb
