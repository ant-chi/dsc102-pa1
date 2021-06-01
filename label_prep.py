from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as f

df = spark.read.option("delimiter", "|").csv("s3://ds102-tophbeifong-scratch/performance_200[0-9]Q[1-4].txt")
# df = spark.read.option("delimiter", "|").csv("s3://ds102-tophbeifong-scratch/performance_sample_2009.txt")
df = df.select("_c0", "_c3", "_c8")

df = df.withColumnRenamed("_c0", "Loan Sequence Number"
      ).withColumnRenamed("_c3", "Current Loan Delinquency Status"
      ).withColumnRenamed("_c8", "Zero Balance Code")

df = df.na.fill("-1")

defaultBalanceCodes = ["03", "06", "09"]
nonDefaultDelinquencyStatus = ["0", "1", "2"]

defaultCond = lambda x, y: 1 if ((x in defaultBalanceCodes) or (y not in nonDefaultDelinquencyStatus)) else 0
labelCond = lambda x: int(bool(x>=1))

dfTemp = df.rdd.map(lambda x: (x["Loan Sequence Number"],x["Current Loan Delinquency Status"],x["Zero Balance Code"], defaultCond(x["Zero Balance Code"], x["Current Loan Delinquency Status"]))).toDF(["Loan Sequence Number", "Current Loan Delinquency Status", "Zero Balance Code", "defaultInstance"])

dfTemp = dfTemp.groupBy('Loan Sequence Number').agg({"defaultInstance": "sum"})
output = dfTemp.rdd.map(lambda x: (x["Loan Sequence Number"],labelCond(x["sum(defaultInstance)"]))).toDF(["pKey", "Label"])

output.write.parquet("s3://ds102-tophbeifong-scratch/sampleLabels.parquet", mode="overwrite")


# columnNames =  spark.read.option("delimiter", ",").text("s3://ds102-tophbeifong-scratch/dfColumns/monthlyPerformanceColumns.txt")
# df.schema.names = columnNames.collect()[0][0].split(",")
# df.schema.names

# f2 = open("dsc102-pa1/monthlyPerformanceColumns.txt", "r")
# monthlyPerformanceColumns = "".join(f2.readlines()).strip("\n").split(",")
# df2.columns = monthlyPerformanceColumns


# aws s3 cp s3://ds102-tophbeifong-scratch/labels_2009Q1_2.parquet/ labels_2009Q1_2.parquet/
