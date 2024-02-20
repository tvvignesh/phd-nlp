from transformers import pipeline

# Load the pipeline for token classification (POS tagging)
pos_tagger = pipeline("ner", model="xlm-roberta-base")

# Example sentences in different languages
sentences = [
    "They are running in the park.",               # English
    "Je regarde un film hier.",                    # French
    "Él camina en el parque.",                     # Spanish
    "Heute ist ein schöner Tag zum Spazierengehen." # German
]

# Tag each example sentence using the multilingual POS tagger
for sentence in sentences:
    # Tag the sentence using the multilingual POS tagger
    tagged_sentence = pos_tagger(sentence)
    
    # Extract words and tags from the tagged sentence
    words = [token['word'] for token in tagged_sentence]
    tags = [token['entity'] for token in tagged_sentence]
    
    print(f"Sentence: {sentence}")
    print("POS Tags:")
    print(list(zip(words, tags)))
    print()
