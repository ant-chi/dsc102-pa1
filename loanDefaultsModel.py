import pyarrow.parquet as pq
import pandas as pd
import s3fs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# client = Client()
s3 = s3fs.S3FileSystem()

labels = pq.ParquetDataset('s3://ds102-tophbeifong-scratch/labels_2009Q1_2.parquet', filesystem=s3).read_pandas().to_pandas()
features = pd.read_parquet("features_2009Q1_2.parquet")

df = pd.merge(labels, features, left_on="pKey", right_on="Loan Sequence Number", how="inner")
df.drop(columns=["Loan Sequence Number"], inplace=True)

trainSplit, testSplit = train_test_split(df, test_size=0.2)
trainFeats, testFeats = trainSplit.iloc[:, 2:], testSplit.iloc[:, 2:]
trainLabels, testLabels = trainSplit["Label"], testSplit["Label"]

grid={"C":[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
mod = LogisticRegression(max_iter=300, class_weight="balanced")
modCV = GridSearchCV(mod, grid, cv=5)
modCV.fit(trainFeats, trainLabels)

modCV.best_estimator_

bestModel = LogisticRegression(max_iter=300, class_weight="balanced", C=0.001).fit(trainFeats, trainLabels)

bestModel.score(trainFeats, trainLabels)
bestModel.score(testFeats, testLabels)
predLabelProbs = bestModel.predict_proba(df.iloc[:, 2:])[:,1]


out = pd.DataFrame(list(zip(df["pKey"], predLabelProbs)), columns=["Loan Sequence Number", "Prob of Default"])

out.to_parquet("s3://ds102-tophbeifong-scratch/predLabelProbs.parquet")
