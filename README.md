# Email-spam-filter-system
A Simple Naïve Bayes Classifier for Email Spam Detection in C++
**Introduction**:
In this article, we will explore a simple implementation of a Naïve Bayes classifier in C++ that is designed to detect spam emails. The code provided demonstrates how to preprocess email data, extract features, and classify emails as either spam or not spam (ham). By the end of this article, you'll have a clear understanding of how the code works and how you can use it for your own email classification tasks.

**Key Concepts:**
Before diving into the code, let's clarify some key concepts:

**Naïve Bayes Classifier**: This is a probabilistic classifier based on Bayes' theorem, assuming independence among predictors. It's particularly effective for text classification tasks like spam detection.


**Feature Extraction**: This involves converting text data into a numerical format that the classifier can understand. In our case, we will convert emails into vectors based on the frequency of words.

**Preprocessing**: This step cleans the email text by removing unwanted characters and converting everything to lowercase, making it easier to analyze.
Code Structure



**The code is structured into several key components:**

**Data Initialization**: A sample dataset of emails labeled as spam or not spam.

**Preprocessing Function**: A function to clean and tokenize the email text.

**Feature Extraction**: A function to create a vocabulary from the dataset.

**Email Vectorization**: A function to convert emails into numerical vectors based on the vocabulary.

**Naïve Bayes Classifier Class**: This class contains methods for training the model and making predictions.

**Main Function**: The entry point of the program where everything comes together.
