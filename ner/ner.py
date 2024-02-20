import spacy

# Load the English language model with named entity recognition (NER) component
nlp = spacy.load("en_core_web_sm")

# Example sentences
sentences = [
    "Barack Obama was the 44th President of the United States.",
    "The Eiffel Tower is located in Paris, France.",
    "Elon Musk is the CEO of SpaceX and Tesla.",
    "Albert Einstein was a German-born theoretical physicist.",
    "The Amazon River is the largest river by discharge volume in the world."
]

# Perform named entity recognition (NER) on each example sentence
for sentence in sentences:
    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Print the named entities detected in the sentence
    print(f"Sentence: {sentence}")
    print("Named Entities:")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Type: {ent.label_}")
    print()