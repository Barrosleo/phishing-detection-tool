# Phishing Detection Tool

This tool detects phishing emails based on various heuristics and machine learning techniques.

## Table of Contents
- Introduction
- Features
- Technologies Used
- Setup Instructions
- Usage
- Contributing
- License

## Introduction
Phishing emails are a common threat in today's digital world. This project leverages machine learning techniques to detect phishing emails based on their content.

## Features
- **Email Content Analysis**: Analyzes the content of emails to detect phishing attempts.
- **Machine Learning**: Uses a Naive Bayes classifier to classify emails as phishing or not.
- **Feature Extraction**: Utilizes TF-IDF for feature extraction from email content.

## Technologies Used
- **Python**: Programming language used for the tool.
- **Machine Learning**: Scikit-learn for building the classifier.
- **Natural Language Processing**: NLTK for text processing.
- **Email Parsing**: Python's email library for parsing email content.

## Setup Instructions
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/phishing-detection-tool.git
    cd phishing-detection-tool
    ```

2. **Install Dependencies**:
    ```bash
    pip install pandas scikit-learn nltk email
    ```

3. **Download NLTK Data**:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4. **Prepare the Dataset**:
    - Download a dataset of emails (e.g., Enron email dataset).
    - Organize the dataset into folders for phishing and non-phishing emails.

## Usage
1. **Run the Python Script**:
    ```bash
    python phishing_detection.py
    ```

2. **Detect Phishing Emails**:
    - Use the `detect_phishing` function to classify email content as phishing or not.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
