import dask.dataframe as dd
from distributed import Client

client=Client()

df1 = dd.read_csv("sample_orig_2009.txt", delimiter="|", header=None, dtype={26:"object", 28:"object"})
df2 = dd.read_csv('sample_svcg_2009.txt', delimiter="|", header = None, dtype=object)


df1.isna().mean(axis=0).compute()
df2.isna().mean(axis=0).compute()

# /home/ubuntu/env/lib/python3.8/site-packages/dask/core.py:151: DtypeWarning: Columns (3,7,23,24,28) have mixed types.Specify dtype option on import or set low_memory=False.
#   result = _execute_task(task, cache)
