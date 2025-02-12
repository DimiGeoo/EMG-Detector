import pandas as pd

file_path = "./MachineLearning/Datasets/dataset.csv"

df = pd.read_csv(file_path)

print("DataFrame Info:")
print(df.info())
print("\nShape of the dataframe: ", df.shape)
