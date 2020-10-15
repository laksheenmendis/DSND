import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)

    X = df['message']
#     y = df[df.columns[5:]]
    y = df[df.columns[4:]]
#     one_hot_encoded = pd.get_dummies(df[['genre','related']])
#     y = pd.concat([one_hot_encoded, y], axis=1)
    
    cat_names = y.columns
    
    return X, y, cat_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    multiOutClF = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', multiOutClF),
    ])
    
    parameters = {
        'clf__estimator__min_samples_leaf': [1, 2, 3]
    #     'clf__estimator__min_samples_split': [2, 3],
    #     'clf__estimator__min_weight_fraction_leaf': [0.0, 0.2, 0.4],
    #     'clf__estimator__n_estimators': [10, 15, 20],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    count = 0
    result_grid = pd.DataFrame(columns=['Category', 'Precision', 'Recall', 'f1-score'])

    # Predict categories of messages.
    Y_pred = model.predict(X_test)
    
    for cat in Y_test.columns:
        #classification_report method returns a dictionary when we set the param output_dict=True
        #but, for some odd reason, it doesn't allow me to do so. (may be this is an old version)
        #hence, I'm getting a string as the result of classification_report function.
        #after doing some string processing, I'm able to retrieve precision, recall and f1-score
        report_str = classification_report(Y_test[cat], Y_pred[:,count])

        result_grid.set_value(count+1, 'Category', cat)

        category_metrics = report_str.split('\n')[-2].split()

        for (index,j) in enumerate(category_metrics):

            if index == 3:
                result_grid.set_value(count+1, 'Precision', category_metrics[index])
            if index == 4:
                result_grid.set_value(count+1, 'Recall', category_metrics[index])
            if index == 5:
                result_grid.set_value(count+1, 'f1-score', category_metrics[index])
        count += 1

    print(result_grid)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()