import pandas as pd
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class CommmentAnalyser():
    def __init__(self, df):
        self.df_main = df
        print(self.df_main.columns)

        #self.save_obj_as_pickle(self.df_main, 'saved_data/df_main_transformed.pkl')
        #self.df_main = self.load_obj_from_pickle('saved_data/df_main.pkl')

        self.prepare_samples()
        self.save_obj_as_pickle(self.cv, 'saved_data/cv.pkl')

        # model_NB =self.get_model(MultinomialNB)
        # print(model_NB)
        #

        # self.best_params_RF = self.optimise_model(RandomForestClassifier, n_estimators=[5], min_samples_leaf=[1,2])
        # model_RF = self.get_model(RandomForestClassifier, **self.best_params_RF)
        #
        # self.save_obj_as_pickle(model_RF, 'saved_data/best_model_RF.pkl')

        # with open('saved_data/best_model_RF', 'rb') as filein:
        #     tmp_model = pickle.load(filein)
        # print(model_RF)
        # print(tmp_model)

    @classmethod
    def from_datafile(cls, filename):
        df = pd.read_csv(filename)
        df = cls.transform_comments(df)

        return cls(df)

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as filein:
            df = pickle.load(filein)
        df = cls.transform_comments(df)

        return cls(df)

    @classmethod
    def from_transformed_pickle(cls, filename):
        with open(filename, 'rb') as filein:
            df = pickle.load(filein)

        return cls(df)

    @classmethod
    def get_wordnet_pos(cls, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    @classmethod
    def transform_comments(cls, df, colname='Tweet'):
        lem = WordNetLemmatizer()
        tokenized_tweets = []
        texts_transformed = []
        for irow, tweet in enumerate(df[colname]):
            tokenized_tweets.append([])
            adjectives = []
            for sentence in nltk.sent_tokenize(tweet):
                tmp_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
                sentence_lemmatized = []
                for word in tmp_tagged:
                    tag = word[1]
                    word_lemmatized = lem.lemmatize(word[0], cls.get_wordnet_pos(tag))
                    sentence_lemmatized.append((word_lemmatized, tag))
                    # if tag == 'JJ' or tag == 'VBG':
                    adjectives.append(word_lemmatized)
                tokenized_tweets[irow].append(sentence_lemmatized)

            texts_transformed.append(' '.join(adjectives))

        df['Tokenized_Tweets'] = tokenized_tweets
        df['Texts_Transformed'] = texts_transformed

        return df

    def save_obj_as_pickle(self, obj, filename):
        with open(filename, 'wb') as fileout:
            pickle.dump(obj, fileout)

    # def load_obj_from_pickle(self, filename):
    #     with open(filename, 'rb') as filein:
    #         obj = pickle.load(filein)
    #     return obj

    def prepare_samples(self, colname='Sentiment'):
        X = self.df_main['Texts_Transformed']
        y = self.df_main[colname]  # == 'positive'

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)

        self.cv = CountVectorizer()
        self.cv.fit(self.X_train)

        self.X_train = self.cv.transform(self.X_train)
        self.X_test = self.cv.transform(self.X_test)

    def get_model(self, estimator, **hyperparams):
        if type(estimator) == str:
            estimator = globals().get(estimator)

        model = estimator(**hyperparams)
        model.fit(self.X_train.toarray(), self.y_train)

        return model

    def optimise_model(self, estimator, **hyperparams):
        if type(estimator) == str:
            estimator = globals().get(estimator)

        gs = GridSearchCV(estimator(), param_grid=hyperparams, cv=5)
        gs.fit(self.X_train, self.y_train)

        opt_params = {key: getattr(gs.best_estimator_,key) for key in hyperparams.keys()}
        print(opt_params)

        return opt_params