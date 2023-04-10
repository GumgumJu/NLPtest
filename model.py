import pandas as pd
import numpy as np
import re
import nltk
from joblib import dump
from nltk.corpus import stopwords
import unicodedata
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# Read file paths and stopwords
PATH = ".\data\extract_normalised_name_fr_training_data.xlsx"
nltk.download('stopwords')

# Convert all accents to standard characters
def remove_accent(s):
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')


# Preprocess a string.
def preprocess_sentence(w):
        # Convert to lowercase
        w = w.casefold().strip()

        # Remove stopwords
        stop_words = set(stopwords.words('french'))

        # replacing everything with space except (a-z, A-Z, 0-9，"'")
        w = re.sub(r"[^a-zA-Z0-9'¿]+", " ", w)
        w = re.sub(r"[']+", "", w)

        # Delete all words with numbers
        pattern = re.compile(r'\b\w*\d\w*\b')
        w = re.sub(pattern, '', w)
        w = re.sub(' +', ' ', w)
        words = w.split()

        filtered_words = [word for word in words if word not in stop_words]
        filtered_text = ' '.join(filtered_words)
        filtered_text = filtered_text.strip()
        return filtered_text


# Get the pre-processed list of designation_fr
def create_dataset(path):
        data = pd.read_excel(path)
        data['designation_fr'] = data.apply(lambda row: preprocess_sentence(remove_accent(row['designation_fr'])),axis=1)
        return data


# Encode each category one-hot and return a dictionary int_to_text for decoding (For RandomForest)
def get_one_hot(data):
        norma_name = data.iloc[:,1].unique()
        text_list = data.iloc[:,1].copy()
        text_to_int = {text:i for i, text in enumerate(norma_name)}
        int_to_text = {i:text for i, text in enumerate(norma_name)}
        labels_encoded = np.array([text_to_int[label] for label in text_list])

        encoder = OneHotEncoder(sparse= False)
        labels_onehot = encoder.fit_transform(labels_encoded.reshape(-1,1))
        return labels_onehot,int_to_text


# Encode each category Int and return a dictionary int_to_text for decoding (For GradientBoosting)
def get_label_encoder(data):
       text_list = data.iloc[:,1].copy()
       label_encoder = LabelEncoder()
       label_encoder.fit(text_list)
       labels = label_encoder.transform(text_list)
       int2text = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
       return labels, int2text


# Calculate the input word vector
def sentence_vector(sentence, Word2Vec_model):
        swords = sentence.split()
        vecs = [Word2Vec_model.wv[word] for word in swords if word in Word2Vec_model.wv]
        vec = np.mean(vecs, axis=0)
        return vec


# Obtain the word vectors of the training set and output the trained Word2Vec model
def get_word2vec(data):
        words = data.iloc[:,0].copy()
        data = [sentence.split() for sentence in words]
        model = Word2Vec(data, min_count=1, vector_size= 50,workers=3, window =3, sg = 1)
        vectors = [sentence_vector(sentence, model) for sentence in words]
        return np.array(vectors), model
        

# Prepare all data for training the model
def prepare_data(PATH):
        data = create_dataset(PATH)
        one_hot_name, onehotint_to_text = get_one_hot(data) 
        word2vec_name, word2vec_model = get_word2vec(data)
        lables, int2text = get_label_encoder(data)
        return one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables, int2text


# Training random forest model with K-fold cross-validation
def RFC_model(X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        pipeline = make_pipeline(StandardScaler(), rf)
        scores = cross_val_score(pipeline, X=X, y=y, cv=10, n_jobs=1)

        print('Cross Validation accuracy scores: %s' % scores)
        print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
        
        rf.fit(X, y)
        return rf


# Training GradientBoosting model with K-fold cross-validation
def GBC_model(X, y):
        clf = GradientBoostingClassifier()
        pipeline = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipeline, X=X, y=y, cv=4, n_jobs=1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X, y)
        print('Cross Validation accuracy scores: %s' % scores)
        print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
        # predicted = clf.predict(X_test)
        # print(y_test)
        # print(classification_report(y_test, predicted))
        return clf



def get_model(PATH):
        one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables, int2text = prepare_data(PATH)
        rf_model = RFC_model(word2vec_name, one_hot_name)
        gbc_model = GBC_model(word2vec_name, lables)
        return rf_model, gbc_model,word2vec_model, onehotint_to_text, int2text

# Save model to the model folder
def save_model(rf_model, gbc_model, word2vec_model, onehotint_to_text, int2text):
        dump(rf_model, "./models/rf_model.sav")
        dump(gbc_model, './models/gbc_model.sav')
        dump(word2vec_model, './models/word2vec_model.sav')
        dump(int2text, './models/int2text.sav')
        dump(onehotint_to_text, './models/onehotint_to_text.sav')


def main():
        rf_model, gbc_model, word2vec_model, onehotint_to_text, int2text = get_model(PATH)
        save_model(rf_model, gbc_model, word2vec_model, onehotint_to_text, int2text)


if __name__ == '__main__':
    main()