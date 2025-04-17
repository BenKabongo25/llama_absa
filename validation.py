# Ben Kabongo
# Statement Extraction and Aspect-Based Sentiment Analysis Validation

# April 2025


from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

def validate_structure(json_output):
    if not isinstance(json_output, list):
        return False
    required_keys = {"statement", "aspect", "sentiment"}
    for entry in json_output:
        if not isinstance(entry, dict):
            return False
        if not required_keys.issubset(entry.keys()):
            return False
    return True

def validate_fields(entry):
    return (
        isinstance(entry["statement"], str)
        and isinstance(entry["aspect"], str)
        and isinstance(entry["sentiment"], str)
        and entry["sentiment"].lower() in VALID_SENTIMENTS
    )

def is_semantically_similar(model, statement, original_sentences, threshold=0.8):
    emb_statement = model.encode(statement, convert_to_tensor=True)
    emb_sentences = model.encode(original_sentences, convert_to_tensor=True)
    sims = util.cos_sim(emb_statement, emb_sentences)
    return bool((sims > threshold).any())

def aspect_in_statement_and_review(aspect, statement, review):
    return aspect.lower() in statement.lower() and aspect.lower() in review.lower()

def check_sentiment_agreement(model, statement, extracted_sentiment):
    result = model(statement)[0]
    return result["label"].lower() == extracted_sentiment.lower()

def main():
    paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    sentiment_model = pipeline("sentiment-analysis")
    #TODO
