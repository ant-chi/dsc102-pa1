import dask.dataframe as dd
from dask import delayed, compute
from distributed import Client
import dask_ml.preprocessing as dPP

client = Client()

df = dd.read_csv("origination_*.txt", delimiter="|", header=None, dtype={25:"object", 26:"object", 28:"object"})
# df = dd.read_csv("origination_2009Q1.txt", delimiter="|", header=None, dtype={25:"object", 26:"object", 28:"object"})
# df = dd.read_csv("sample_orig_2009.txt", delimiter="|", header=None, dtype={26:"object", 28:"object"}


f = open("dsc102-pa1/originationColumns.txt", "r")
originationColumns = "".join(f.readlines()).strip("\n").split(",")
df.columns = originationColumns

df.isna().mean(axis=0).compute()
# df2.isna().mean(axis=0).compute()
#
# defaultBalanceCodes = ["03", "06", "09"]
# nonDefaultDelinquencyStatus = ["0", "1", "2"] # doublecheck for XX, R, and Space
# #
# def genLabels(x):
#     """
#     Determines if a loan can be considered a default. A loan is considered a default if it is more
#     than 90 days due or if any of the following conditions apply: short sale or charge off,
#     repurchase prior to property disposition, or REO disposition.
#     """
#     if (x["Current Loan Delinquency Status"] not in nonDefaultDelinquencyStatus) or (x["Zero Balance Code"] in defaultBalanceCodes):
#         return 1
#     else:
#         return 0
#
# df2["Loan Sequence Number"] = df2["Loan Sequence Number"].astype(str)
# #
# df2["Sublabels"] = df2.apply(genLabels, axis=1)
# labels = df2.groupby("Loan Sequence Number")["Sublabels"].sum().compute()

# Cast types to category for OneHotEncoder
df["Loan Purpose"] = df["Loan Purpose"].astype("category").cat.as_known()
df = df.assign(id=(df["Loan Purpose"].cat.codes))

df["First Time Homebuyer Flag"] = df["First Time Homebuyer Flag"].astype("category").cat.as_known()
df = df.assign(id=(df["First Time Homebuyer Flag"].cat.codes))

df["Property Type"] = df["Property Type"].astype("category").cat.as_known()
df = df.assign(id=(df["Property Type"].cat.codes))

df["Occupancy Status"] = df["Occupancy Status"].astype("category").cat.as_known()
df = df.assign(id=(df["Occupancy Status"].cat.codes))
df.compute()

ohe = dPP.OneHotEncoder(sparse=False)
ohe.fit(df[["Loan Purpose", "First Time Homebuyer Flag", "Property Type", "Occupancy Status"]])
features = ohe.transform(df[["Loan Purpose", "First Time Homebuyer Flag", "Property Type", "Occupancy Status"]]).compute()

# binarize based on FRM (Y/N)
features["Is FRM"] = (df["Amortization Type"]=="FRM").astype(int)

# Impute unknown values with mean
meanValues = df[["Credit Score", "DTI Ratio", "CLTV"]].mean().compute()
features["Credit Score"] = df["Credit Score"].replace({9999: meanValues["Credit Score"]}).compute()
features["DTI Ratio"] = df["DTI Ratio"].replace({999: meanValues["DTI Ratio"]}).compute()
features["CLTV"] = df["CLTV"].replace({999: meanValues["CLTV"]}).compute()

standardizer = dPP.StandardScaler()
standardizer.fit(features[["Credit Score", "DTI Ratio", "CLTV"]])
features[["Credit Score", "DTI Ratio", "CLTV"]] = standardizer.transform(features[["Credit Score", "DTI Ratio", "CLTV"]])

features["Loan Sequence Number"] = df["Loan Sequence Number"]

features.to_parquet(path='s3://ds102-tophbeifong-scratch/features_2009Q1_2.parquet')
