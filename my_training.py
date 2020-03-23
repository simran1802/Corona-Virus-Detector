import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(40)
    shuffle = np.random.permutation(len(data))
    test_size = int(len(data)*ratio)
    test_indices = shuffle[:test_size]
    train_indices = shuffle[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    # Load the data
    df = pd.read_excel("carona_data.xlsx")
    train,test = data_split(df,0.2)

    # Train-test split
    x_train = train[['fever','body_pain','runny_nose','diff_breath']].to_numpy()
    x_test = test[['fever','body_pain','runny_nose','diff_breath']].to_numpy()

    y_train = train[['infection_problem']].to_numpy().reshape(1894,)
    y_test = test[['infection_problem']].to_numpy().reshape(473,)

    clf = LogisticRegression()
    clf.fit(x_train,y_train)

    file = open('model.pkl','wb')

    pickle.dump(clf,file)
    file.close()


    
