import dask.dataframe as dd
from distributed import Client
import dask_ml.preprocessing as dPP
# from dask_ml.preprocessing import OneHotEncoder

client = Client()

df1 = dd.read_csv("sample_orig_2009.txt", delimiter="|", header=None, dtype={26:"object", 28:"object"})
df2 = dd.read_csv('sample_svcg_2009.txt', delimiter="|", header = None, dtype=object)

f1 = open("dsc102-pa1/originationColumns.txt", "r")
f2 = open("dsc102-pa1/monthlyPerformanceColumns.txt", "r")

originationColumns = "".join(f1.readlines()).strip("\n").split(",")
monthlyPerformanceColumns = "".join(f2.readlines()).strip("\n").split(",")

df1.columns = originationColumns
df2.columns = monthlyPerformanceColumns

df1.isna().mean(axis=0).compute()
df2.isna().mean(axis=0).compute()

defaultBalanceCodes = ["03", "06", "09"]
nonDefaultDelinquencyStatus = ["0", "1", "2"] # doublecheck for XX, R, and Space

def genLabels(x):
    """
    Determines if a loan can be considered a default. A loan is considered a default if it is more
    than 90 days due or if any of the following conditions apply: short sale or charge off,
    repurchase prior to property disposition, or REO disposition.
    """
    if (x["Current Loan Delinquency Status"] not in nonDefaultDelinquencyStatus) or (x["Zero Balance Code"] in defaultBalanceCodes):
        return 1
    else:
        return 0

df2["Loan Sequence Number"] = df2["Loan Sequence Number"].astype(str)

df2["Sublabels"] = df2.apply(genLabels, axis=1)
labels = df2.groupby("Loan Sequence Number")["Sublabels"].sum().compute()

# Prep "Loan Purpose", "First Time Homebuyer Flag", "Property Type", and "Occupancy Status" for OHE
df1["Loan Purpose"] = df1["Loan Purpose"].astype("category")
df1["Loan Purpose"] = df1["Loan Purpose"].cat.as_known()
df1 = df1.assign(id=(df1["Loan Purpose"].cat.codes))

df1["First Time Homebuyer Flag"] = df1["First Time Homebuyer Flag"].astype("category")
df1["First Time Homebuyer Flag"] = df1["First Time Homebuyer Flag"].cat.as_known()
df1 = df1.assign(id=(df1["First Time Homebuyer Flag"].cat.codes))

df1["Property Type"] = df1["Property Type"].astype("category")
df1["Property Type"] = df1["Property Type"].cat.as_known()
df1 = df1.assign(id=(df1["Property Type"].cat.codes))

df1["Occupancy Status"] = df1["Occupancy Status"].astype("category")
df1["Occupancy Status"] = df1["Occupancy Status"].cat.as_known()
df1 = df1.assign(id=(df1["Occupancy Status"].cat.codes))
df1.compute()

ohe = dPP.OneHotEncoder(sparse=False)
ohe.fit(df1[["Loan Purpose", "First Time Homebuyer Flag", "Property Type", "Occupancy Status"]])
features = ohe.transform(df1[["Loan Purpose", "First Time Homebuyer Flag", "Property Type", "Occupancy Status"]]).compute()


features["Is FRM"] = (df1["Amortization Type"]=="FRM").astype(int)

meanValues = df1[["Credit Score", "DTI Ratio", "CLTV"]].mean().compute()

features["Credit Score"] = df1["Credit Score"].replace({9999: meanValues["Credit Score"]})#.compute()
features["DTI Ratio"] = df1["DTI Ratio"].replace({999: meanValues["DTI Ratio"]})#.compute()
features["CLTV"] = df1["CLTV"].replace({999: meanValues["CLTV"]})#.compute()


standardizer = dPP.StandardScaler()
standardizer.fit(features[["Credit Score", "DTI Ratio", "CLTV"]])
features[["Credit Score", "DTI Ratio", "CLTV"]] = standardizer.transform(features[["Credit Score", "DTI Ratio", "CLTV"]])

features["Loan Sequence Number"] = df1["Loan Sequence Number"]
features.to_parquet(path='s3://ds102-tophbeifong-scratch/features.parquet')

# list(df2[df2["Loan Sequence Number"]=="F09Q10000078"]["Current Loan Delinquency Status"].compute())
#
# list(df2[df2["Loan Sequence Number"]=="F09Q10000078"]["Current Loan Delinquency Status"].compute())
# list(df2[df2["Loan Sequence Number"]=="F09Q10000013"]["Zero Balance Code"].value_counts(dropna=False).compute())
# list(df2[df2["Loan Sequence Number"]=="F09Q10000078"]["Zero Balance Code"].compute())

# df2[df2["Loan Sequence Number"]=="F09Q10000078"].apply(genLabels, axis=1).sum().compute()
