import pandas as pd
df = pd.read_csv('data/kinshiphinton/kinship_hinton_qa_2hop.csv')
unique_liquid_values = df['SplitLabel'].unique()
print(unique_liquid_values)

