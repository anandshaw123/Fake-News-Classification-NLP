import timeit
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import _pickle as pickle
import string
import nltk
nltk.data.path.append('./data/nltk_data')

start = timeit.default_timer()
with open("data/model.pkl", 'rb') as f:
        pipeline = pickle.load(f)
        stop = timeit.default_timer()
        print('Model Loaded in: ', stop - start)


class Predict:
    output = {}

    def __init__(self, text):
        self.output['original'] = text

    def predict(self):

        self.preprocess()
        self.pos_tag_words()

        clean_and_pos_tagged_text = self.output['preprocessed'] + \
            ' ' + self.output['pos_tagged']

        self.output['prediction'] = 'FAKE' if pipeline.predict(
            [clean_and_pos_tagged_text])[0] == 0 else 'REAL'

        return self.output

    def preprocess(self):
        text = self.output['original'].lower()
        text = [t for t in text.split(" ") if len(t) > 1]
        text = [word for word in text if not any(c.isdigit() for c in word)]
        text = [word.strip(string.punctuation) for word in text]
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        text = [t for t in text if len(t) > 0]
        pos_tags = pos_tag(text)
        text = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1]))
                for t in pos_tags]
        self.output['preprocessed'] = " ".join(text)

    def get_wordnet_pos(self, pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def pos_tag_words(self):
        pos_text = nltk.pos_tag(
            nltk.word_tokenize(self.output['preprocessed']))
        self.output['pos_tagged'] = " ".join(
            [pos + "-" + word for word, pos in pos_text])