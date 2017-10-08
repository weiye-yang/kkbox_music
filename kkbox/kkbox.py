import numpy as np
import pandas as pd
from sklearn import model_selection
import tqdm
import lightgbm as lgbm
import gc

members_read = pd.read_csv("members.csv")
songs_read = pd.read_csv("songs.csv")
train_read = pd.read_csv("train.csv")
test_read = pd.read_csv("test.csv")

#format member data, in particular dates
members_read["registration_year"] = members_read["registration_init_time"].apply(lambda i: int(str(i)[0:4]))
members_read["registration_month"] = members_read["registration_init_time"].apply(lambda i: int(str(i)[4:6]))
del members_read["registration_init_time"]
members_read["expiration_year"] = members_read["expiration_date"].apply(lambda i: int(str(i)[0:4]))
members_read["expiration_month"] = members_read["expiration_date"].apply(lambda i: int(str(i)[4:6]))
del members_read["expiration_date"]

#format song data
del songs_read["composer"], songs_read["lyricist"]
#the training and test data only use a portion of the songs so we'll do the difficult formatting later

#combine all relevant data
train_all = train_read[pd.notnull(train_read["target"])]
train_all = train_all.merge(songs_read, on="song_id", how="left")
test_all = test_read.merge(songs_read, on="song_id", how="left")
train_all = train_all.merge(members_read, on="msno", how="left")
test_all = test_all.merge(members_read, on="msno", how="left")
#fill in missing data
train_all = train_all.fillna(-1)
test_all = test_all.fillna(-1)

songs_used = songs_read[["song_id", "genre_ids"]]
songs_used = songs_used[songs_used["song_id"].isin(pd.concat([train_all["song_id"], test_all["song_id"]]))] #songs used in the data
del train_read, test_read, members_read, songs_read
gc.collect()

#format genres into categories - one song may belong to multiple genres
songs_used["genre_ids"].values[pd.isnull(songs_used["genre_ids"])] = ""
songs_used["genre_set"] = songs_used["genre_ids"].apply(lambda st: set(st.split("|")))
del songs_used["genre_ids"]
all_genres = set.union(*songs_used[songs_used["song_id"].isin(train_all["song_id"])]["genre_set"]) #genres appearing in training set
all_genres.remove("")
for genre in tqdm.tqdm(all_genres):
    songs_used["genre_" + genre] = songs_used["genre_set"].apply(lambda se: genre in se)
del songs_used["genre_set"]

#final merge
train_all = train_all.merge(songs_used, on="song_id", how="left")
test_all = test_all.merge(songs_used, on="song_id", how="left")
del train_all["genre_ids"], test_all["genre_ids"]
gc.collect()

#fill in missing genre data
train_all = train_all.fillna(False)
test_all = test_all.fillna(False)

#convert integer categorical data into strings
int2str = ["city", "bd", "registered_via", "language"]
for column in int2str:
    train_all[column] = train_all[column].apply(str)
    test_all[column] = test_all[column].apply(str)
#convert string data into categorical
columns = list(train_all.columns)
columns.remove("target")
for column in columns:
    if train_all[column].dtype == "object":
        train_all[column] = train_all[column].astype("category", ordered = False)
        test_all[column] = test_all[column].astype("category", ordered = False)

X_train = train_all.copy()
del X_train["target"]
y_train = train_all["target"]
X_test = test_all.copy()
del X_test["id"]
ids = np.array(test_all["id"])

#split train into training and validation sets
num_folds = 3
kf = model_selection.KFold(n_splits = num_folds)
param = {}
param["metric"] = "auc"
param["application"] = "binary"
param["num_leaves"] = 128
pred = np.zeros(shape=[len(test_all)])
#fit lgbm
for ind_tr, ind_va in kf.split(X_train):
    data_tr = lgbm.Dataset(X_train.ix[ind_tr], label = y_train[ind_tr])
    data_va = lgbm.Dataset(X_train.ix[ind_va], label = y_train[ind_va])
    light = lgbm.train(param,data_tr, valid_sets = data_va, num_boost_round = 150)

    #predict
    pred += light.predict(X_test)

predictions = pred / num_folds
#write to file
output = pd.DataFrame(ids, columns = ["id"])
output["target"] = predictions

#output.to_csv("prediction.csv", index = False)
