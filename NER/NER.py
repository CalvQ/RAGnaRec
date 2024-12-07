import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
import re

# TODO: look at lemmatizing using spaCy


class NER():
    def __init__(self, TARGET_TAGS=["FAC", "ORG", "PERSON"], REPLACEMENT_TAGS=["FAC", "ORG", "PERSON"]):
        self.TARGET_TAGS = TARGET_TAGS
        self.REPLACEMENT_TAGS = REPLACEMENT_TAGS
        self.nlp = spacy.load("en_core_web_trf")
        self.tag_list = "|".join([label.lower()
                                 for label in self.REPLACEMENT_TAGS])
        self.token_regex = fr"(?:\b\w+\b|<(?:{self.tag_list})>)"

        self.tokenizer = RegexpTokenizer(self.token_regex)
        self.lemmatizer = WordNetLemmatizer()

        self.english_stopwords = set(nltk.corpus.stopwords.words('english'))

    def replace_specific_entities(self, doc, replacement_tags=None):
        if not replacement_tags:
            replacement_tags = self.REPLACEMENT_TAGS

        replaced_text = doc.text
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            if ent.label_ in replacement_tags:
                token = f"<{ent.label_}>"
                replaced_text = replaced_text[:ent.start_char] + \
                    token + replaced_text[ent.end_char:]
        return replaced_text

    def tag_review(self, review, target_tags=None, replacement_tags=None):
        if not target_tags:
            target_tags = self.TARGET_TAGS

        doc = self.nlp(review)
        contain_flag = False
        for ent in doc.ents:
            if ent.label_ in target_tags:
                contain_flag = True
                break

        result = self.replace_specific_entities(
            doc, replacement_tags=replacement_tags)

        return contain_flag, result

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'n'

    def clean_tokens(self, tokens):
        output = []
        pos_tags = nltk.pos_tag(tokens)
        for token in pos_tags:
            lemma = self.lemmatizer.lemmatize(
                token[0], pos=self._get_wordnet_pos(token[1]))
            if lemma not in self.english_stopwords and len(lemma) >= 2:
                output.append(lemma)
        return output

    def preprocess_text(self, review, target_tags=None, replacement_tags=None):
        contain_flag, tagged_review = self.tag_review(
            review, target_tags, replacement_tags)
        text = tagged_review.lower()
        text = re.sub(r"[^\w\s<>/]", "", text)
        tokens = self.tokenizer.tokenize(text)
        cleaned_tokens = self.clean_tokens(tokens)
        return contain_flag, cleaned_tokens

    def preprocess_review(self, review, target_tags=None, replacement_tags=None):
        _, result = self.preprocess_text(review, target_tags, replacement_tags)
        return result
