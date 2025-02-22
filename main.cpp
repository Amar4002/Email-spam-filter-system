#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <algorithm>
using namespace std;

// Sample email dataset
vector<pair<string, string>> data = {
    {"Win a lottery now!", "spam"},
    {"Meeting at 5 pm", "not_spam"},
    {"Earn money from home", "spam"},
    {"Project update required", "not_spam"},
    {"Get free rewards now", "spam"},
    {"Let's catch up tomorrow", "not_spam"}
};

// Preprocess email text
vector<string> preprocess(const string &text) {
    string cleaned;
    for (char c : text) {
        if (isalpha(c) || c == ' ') cleaned += tolower(c);
    }
    
    vector<string> words;
    stringstream ss(cleaned);
    string word;
    while (ss >> word) words.push_back(word);
    return words;
}

// Extract vocabulary from dataset
unordered_map<string, int> extract_features(const vector<pair<string, string>> &emails) {
    unordered_map<string, int> vocab;
    int index = 0;
    for (const auto &email : emails) {
        for (const string &word : preprocess(email.first)) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = index++;
            }
        }
    }
    return vocab;
}

unordered_map<string, int> vocab = extract_features(data);

vector<int> email_to_vector(const string &email) {
    vector<int> vec(vocab.size(), 0);
    for (const string &word : preprocess(email)) {
        if (vocab.find(word) != vocab.end()) {
            vec[vocab[word]]++;
        }
    }
    return vec;
}

// Naïve Bayes Classifier
class NaiveBayes {
private:
    double spam_prob;
    unordered_map<string, double> word_probs_spam, word_probs_ham;
    
public:
    void fit(const vector<vector<int>> &X, const vector<int> &y) {
        int spam_count = count(y.begin(), y.end(), 1);
        int ham_count = y.size() - spam_count;
        spam_prob = (double)spam_count / y.size();
        
        unordered_map<int, int> word_counts_spam, word_counts_ham;
        for (size_t i = 0; i < y.size(); ++i) {
            for (size_t j = 0; j < X[i].size(); ++j) {
                if (y[i] == 1) word_counts_spam[j] += X[i][j];
                else word_counts_ham[j] += X[i][j];
            }
        }
        
        for (const auto &pair : vocab) {
            int word_index = pair.second;
            word_probs_spam[pair.first] = log((word_counts_spam[word_index] + 1.0) / (spam_count + vocab.size()));
            word_probs_ham[pair.first] = log((word_counts_ham[word_index] + 1.0) / (ham_count + vocab.size()));
        }
    }
    
    int predict(const vector<int> &x) {
        double spam_score = log(spam_prob);
        double ham_score = log(1 - spam_prob);
        for (const auto &pair : vocab) {
            int word_index = pair.second;
            if (x[word_index] > 0) {
                spam_score += word_probs_spam[pair.first] * x[word_index];
                ham_score += word_probs_ham[pair.first] * x[word_index];
            }
        }
        return spam_score > ham_score ? 1 : 0;
    }
};

int main() {
    vector<vector<int>> X;
    vector<int> y;
    for (const auto &email : data) {
        X.push_back(email_to_vector(email.first));
        y.push_back(email.second == "spam" ? 1 : 0);
    }
    
    NaiveBayes nb;
    nb.fit(X, y);
    
    vector<string> test_emails = {"Free money now!", "See you tomorrow"};
    for (const string &test : test_emails) {
        vector<int> test_vec = email_to_vector(test);
        cout << "Email: " << test << " -> " << (nb.predict(test_vec) ? "Spam" : "Ham") << endl;
    }
    
    return 0;
}
