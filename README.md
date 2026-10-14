🇮🇳 Indian Language Identification Using NLP

A Machine Learning-based application that identifies Indian languages from text using Character-level TF-IDF and Multinomial Naive Bayes. The project provides an interactive web interface built with Streamlit.

📌 Project Overview

Language identification is an important Natural Language Processing (NLP) task that determines the language of a given text.

This project identifies text written in six Indian languages using a trained Machine Learning model. Users can enter a sentence through the Streamlit interface and view the predicted language along with a confidence estimate.

✨ Features

Identifies six Indian languages.

Uses character-level TF-IDF for text feature extraction.

Uses Multinomial Naive Bayes for classification.

Provides an interactive Streamlit interface.

Displays the predicted language and model confidence estimate.

Includes writing-script validation for user input.

Displays model test accuracy.

🌐 Supported Languages

Hindi

Marathi

Bengali

Gujarati

Tamil

Telugu

🛠️ Technologies Used

Python – Programming language

Pandas – Dataset loading and data preparation

Scikit-learn – Feature extraction, model training, and evaluation

TF-IDF Vectorizer – Character-level text representation

Multinomial Naive Bayes – Language classification

Streamlit – Web application interface

📂 Project Structure

Indian-Language-Identification/
│
├── app.py
├── indian_languages.csv
├── requirements.txt
└── README.md

Update this structure if your repository contains additional files or uses different filenames.

📊 Dataset

The project uses indian_languages.csv, containing 100 text samples labeled with their corresponding languages.

Property

Details

Total samples

100

Number of languages

6

Input column

Text

Target column

Language

File format

CSV

The dataset contains samples from Hindi, Marathi, Bengali, Gujarati, Tamil, and Telugu.

⚙️ Machine Learning Methodology

Load the dataset and prepare the text and language labels.

Split the dataset into training and testing sets using an 80:20 ratio.

Extract character-level TF-IDF features using character n-grams of sizes 1 to 3.

Train a Multinomial Naive Bayes classifier.

Evaluate the model using test accuracy.

Use Streamlit to accept text and display the predicted language.

📈 Model Performance

The application displays a test accuracy of 100.00% on the current held-out test split.

Note: The dataset contains only 100 samples. This result applies to the current test split and does not guarantee the same accuracy on unseen text from other sources.

💻 Installation and Usage

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Indian-Language-Identification

Replace <YOUR_GITHUB_REPOSITORY_URL> with your actual GitHub repository URL.

2. Install Dependencies

pip install -r requirements.txt

3. Run the Application

streamlit run app.py

The application will open in your browser. Enter a sentence in a supported language and click Identify Language to view the prediction.

🧪 Example

Input:

मला मराठी भाषा आवडते।

Expected language label: Marathi

The model also displays a confidence estimate for the prediction. Actual results may vary depending on the input text.

🔮 Future Improvements

Expand the dataset with more text samples.

Add support for additional Indian languages.

Compare different Machine Learning algorithms.

Evaluate performance on larger, independent datasets.

Improve handling of short and mixed-language inputs.

👩‍💻 Author  
Prachi mahesh Dhamode

Project: Indian Language Identification Using NLP
