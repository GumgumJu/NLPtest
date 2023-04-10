from model import preprocess_sentence, remove_accent, sentence_vector
import numpy as np
from joblib import load


def main():

    # load models
    rf_model = load('./models/rf_model.sav')
    gbc_model = load('./models/gbc_model.sav')
    word2vec_model = load('./models/word2vec_model.sav')
    onehotint_to_text = load('./models/onehotint_to_text.sav')
    int2text = load('./models/int2text.sav')

    # Enter the name to be normalized and predict
    predict_name = input("Input a name:  ")
    name_vec = preprocess_sentence(remove_accent(predict_name))
    words = name_vec.split()
    words = [word for word in words if word in word2vec_model.wv.index_to_key]
    filtered_text = ' '.join(words)
    input_data = sentence_vector(filtered_text, word2vec_model).reshape(1,-1)


    try:
        rf_name_class = np.argmax(rf_model.predict(input_data))
        rf_class_name = onehotint_to_text[rf_name_class]
        rf_proba = rf_model.predict_proba(input_data)[rf_name_class][0][1]
        print("predict by RandomForest: ", rf_class_name, rf_proba)

        gbc_name_class = gbc_model.predict(input_data)
        gbc_class_name = int2text[int(gbc_name_class)]
        gbc_proba = np.max(gbc_model.predict_proba(input_data))
        print("predict by GradientBoostingClassifier: ", gbc_class_name, gbc_proba)
        if gbc_proba > rf_proba:
            return (gbc_class_name, gbc_proba)
        else:
            return (rf_class_name, rf_proba)
    
    except ValueError as e:
        print("Cannot predict")
        return None

if __name__ == '__main__':
    main()