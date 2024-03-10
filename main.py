from class_comment_analyser import CommmentAnalyser

if __name__ == '__main__':
    ca = CommmentAnalyser.from_datafile('dataset/fifa_world_cup_2022_tweets.csv')
    #ca = CommmentAnalyser.from_pickle('saved_data/df_main.pkl')
    #ca = CommmentAnalyser.from_transformed_pickle('saved_data/df_main_transformed.pkl')

    best_params_RF = ca.optimise_model('RandomForestClassifier', n_estimators=[5], min_samples_leaf=[1,2])
    model_RF = ca.get_model('RandomForestClassifier', **best_params_RF)
    ca.save_obj_as_pickle(model_RF, 'saved_data/best_model_RF')