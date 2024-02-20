import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Load the Treebank corpus for training
train_data = nltk.corpus.treebank.tagged_sents()[:3000]  # Use first 3000 sentences for training

# Function to extract features from tagged sentences
def extract_features(tagged_sentence, index):
    word, pos_tag = tagged_sentence[index]
    prev_word = '' if index == 0 else tagged_sentence[index-1][0]
    next_word = '' if index == len(tagged_sentence)-1 else tagged_sentence[index+1][0]
    return {'word': word, 'prev_word': prev_word, 'next_word': next_word}

# Extract features and labels from the training data
X_train = []
y_train = []
for sentence in train_data:
    for i in range(len(sentence)):
        X_train.append(extract_features(sentence, i))
        y_train.append(sentence[i][1])

# Define a pipeline for feature extraction, vectorization, and k-nearest neighbors classification
pipeline = Pipeline([
    ('vectorizer', DictVectorizer()),
    ('classifier', KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine'))
])

# Train the memory-based tagger
pipeline.fit(X_train, y_train)

# Example sentences
sentences = [
    "They are running in the park.",
    "I watched a movie yesterday.",
    "He loves ice creams.",
    "She could sing very well.",
    "The cat's toy is missing.",
    "I have two dogs."
]

# Tag each example sentence using the memory-based tagger
for sentence in sentences:
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Extract features for each token
    X_test = [extract_features(list(zip(tokens, [''] * len(tokens))), i) for i in range(len(tokens))]
    
    # Predict POS tags using the trained memory-based tagger
    pos_tags = pipeline.predict(X_test)
    
    # Combine tokens with predicted POS tags
    tagged_sentence = list(zip(tokens, pos_tags))
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(tagged_sentence)
    print()
