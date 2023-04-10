<<<<<<< HEAD
import pandas as pd
import numpy as np
import re
import unicodedata
import tensorflow_hub as hub
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import wget
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout

data = pd.read_excel(".\data\extract_normalised_name_fr_training_data.xlsx")
# url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
# filename = wget.download(url)
nltk.download('stopwords')

PATH = ".\data\extract_normalised_name_fr_training_data.xlsx"

def remove_accent(s):
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')

def preprocess_sentence(w):
        w = w.casefold().strip()
        stop_words = set(stopwords.words('french'))
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z0-9'¿]+", " ", w)
        w = re.sub(r"[']+", "", w)
        pattern = re.compile(r'\b\w*\d\w*\b')
        w = re.sub(pattern, '', w)
        w = re.sub(' +', ' ', w)
        words = w.split()

        filtered_words = [word for word in words if word not in stop_words]
        filtered_text = ' '.join(filtered_words)
        filtered_text = filtered_text.strip()
        return filtered_text


def create_dataset(path):
        # path : path xlsx file
        data = pd.read_excel(path)
        data['designation_fr'] = data.apply(lambda row: preprocess_sentence(remove_accent(row['designation_fr'])),axis=1)
        return data


def get_one_hot(data):
        norma_name = data.iloc[:,1].unique()
        text_list = data.iloc[:,1].copy()
        # type_num = len(norma_name)
        text_to_int = {text:i for i, text in enumerate(norma_name)}
        int_to_text = {i:text for i, text in enumerate(norma_name)}
        labels_encoded = np.array([text_to_int[label] for label in text_list])

        encoder = OneHotEncoder(sparse= False)
        labels_onehot = encoder.fit_transform(labels_encoded.reshape(-1,1))
        return labels_onehot,int_to_text


def get_label_encoder(data):
       text_list = data.iloc[:,1].copy()
       label_encoder = LabelEncoder()
       label_encoder.fit(text_list)
       labels = label_encoder.transform(text_list)
       return labels


def sentence_vector(sentence, Word2Vec_model):
        swords = sentence.split()
        vecs = [Word2Vec_model.wv[word] for word in swords if word in Word2Vec_model.wv]
        vec = np.mean(vecs, axis=0)
        return vec


def get_word2vec(data):
        words = data.iloc[:,0].copy()
        data = [sentence.split() for sentence in words]
        model = Word2Vec(data, min_count=1, vector_size= 50,workers=3, window =3, sg = 1)

        # def sentence_vector(sentence, Word2Vec_model):
        #     swords = sentence.split()
        #     vecs = [Word2Vec_model.wv[word] for word in swords if word in Word2Vec_model.wv]
        #     vec = np.mean(vecs, axis=0)
        #     return vec
        
        vectors = [sentence_vector(sentence, model) for sentence in words]
        return np.array(vectors), model


def prepare_data(PATH):
        data = create_dataset(PATH)
        one_hot_name, onehotint_to_text = get_one_hot(data) 
        word2vec_name, word2vec_model = get_word2vec(data)
        lables = get_label_encoder(data)
        return one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables


def RFC_model(X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        pipeline = make_pipeline(StandardScaler(), rf)
        scores = cross_val_score(pipeline, X=X, y=y, cv=10, n_jobs=1)

        print('Cross Validation accuracy scores: %s' % scores)
        print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
        
        rf.fit(X, y)
        # y_pred = rf.predict(X_test)
        # print("Accuracy:", accuracy_score(y_test, y_pred))
        return rf


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


one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables = prepare_data(PATH)
rf_model = RFC_model(word2vec_name, one_hot_name)
gbc_model = GBC_model(word2vec_name, lables)

predict_name = input("Input a name:  ")
name_vec = preprocess_sentence(remove_accent(predict_name))
input_data = sentence_vector(name_vec, word2vec_model).reshape(1,-1)
try:
        name_class = np.argmax(rf_model.predict(input_data))
        print(onehotint_to_text[name_class])
        print(rf_model.predict_proba(input_data)[name_class][0][0])
except ValueError as e:
        print("Cannot predict")





# model = Sequential()
# model.add(Embedding(input_dim=len(word2vec_model.wv.index_to_key), output_dim=50, weights=[word2vec_model.wv.vectors], input_length=50, trainable=False))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=15, activation='softmax'))
# model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

=======
import pandas as pd
import numpy as np
import re
import unicodedata
import tensorflow_hub as hub
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import wget
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout

data = pd.read_excel(".\data\extract_normalised_name_fr_training_data.xlsx")
# url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
# filename = wget.download(url)
nltk.download('stopwords')

PATH = ".\data\extract_normalised_name_fr_training_data.xlsx"

def remove_accent(s):
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')

def preprocess_sentence(w):
        w = w.casefold().strip()
        stop_words = set(stopwords.words('french'))
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z0-9'¿]+", " ", w)
        w = re.sub(r"[']+", "", w)
        pattern = re.compile(r'\b\w*\d\w*\b')
        w = re.sub(pattern, '', w)
        w = re.sub(' +', ' ', w)
        words = w.split()

        filtered_words = [word for word in words if word not in stop_words]
        filtered_text = ' '.join(filtered_words)
        filtered_text = filtered_text.strip()
        return filtered_text


def create_dataset(path):
        # path : path xlsx file
        data = pd.read_excel(path)
        data['designation_fr'] = data.apply(lambda row: preprocess_sentence(remove_accent(row['designation_fr'])),axis=1)
        return data


def get_one_hot(data):
        norma_name = data.iloc[:,1].unique()
        text_list = data.iloc[:,1].copy()
        # type_num = len(norma_name)
        text_to_int = {text:i for i, text in enumerate(norma_name)}
        int_to_text = {i:text for i, text in enumerate(norma_name)}
        labels_encoded = np.array([text_to_int[label] for label in text_list])

        encoder = OneHotEncoder(sparse= False)
        labels_onehot = encoder.fit_transform(labels_encoded.reshape(-1,1))
        return labels_onehot,int_to_text


def get_label_encoder(data):
       text_list = data.iloc[:,1].copy()
       label_encoder = LabelEncoder()
       label_encoder.fit(text_list)
       labels = label_encoder.transform(text_list)
       return labels


def sentence_vector(sentence, Word2Vec_model):
        swords = sentence.split()
        vecs = [Word2Vec_model.wv[word] for word in swords if word in Word2Vec_model.wv]
        vec = np.mean(vecs, axis=0)
        return vec


def get_word2vec(data):
        words = data.iloc[:,0].copy()
        data = [sentence.split() for sentence in words]
        model = Word2Vec(data, min_count=1, vector_size= 50,workers=3, window =3, sg = 1)

        # def sentence_vector(sentence, Word2Vec_model):
        #     swords = sentence.split()
        #     vecs = [Word2Vec_model.wv[word] for word in swords if word in Word2Vec_model.wv]
        #     vec = np.mean(vecs, axis=0)
        #     return vec
        
        vectors = [sentence_vector(sentence, model) for sentence in words]
        return np.array(vectors), model


def prepare_data(PATH):
        data = create_dataset(PATH)
        one_hot_name, onehotint_to_text = get_one_hot(data) 
        word2vec_name, word2vec_model = get_word2vec(data)
        lables = get_label_encoder(data)
        return one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables


def RFC_model(X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        pipeline = make_pipeline(StandardScaler(), rf)
        scores = cross_val_score(pipeline, X=X, y=y, cv=10, n_jobs=1)

        print('Cross Validation accuracy scores: %s' % scores)
        print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
        
        rf.fit(X, y)
        # y_pred = rf.predict(X_test)
        # print("Accuracy:", accuracy_score(y_test, y_pred))
        return rf


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


one_hot_name, onehotint_to_text, word2vec_name, word2vec_model, lables = prepare_data(PATH)
rf_model = RFC_model(word2vec_name, one_hot_name)
gbc_model = GBC_model(word2vec_name, lables)

predict_name = input("Input a name:  ")
name_vec = preprocess_sentence(remove_accent(predict_name))
input_data = sentence_vector(name_vec, word2vec_model).reshape(1,-1)
try:
        name_class = np.argmax(rf_model.predict(input_data))
        print(onehotint_to_text[name_class])
        print(rf_model.predict_proba(input_data)[name_class][0][0])
except ValueError as e:
        print("Cannot predict")





# model = Sequential()
# model.add(Embedding(input_dim=len(word2vec_model.wv.index_to_key), output_dim=50, weights=[word2vec_model.wv.vectors], input_length=50, trainable=False))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=15, activation='softmax'))
# model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

>>>>>>> a353fe4849db489fedd7dfac6fec34a2f7b06976
# print(len(create_dataset(PATH).iloc[:,1].unique()))