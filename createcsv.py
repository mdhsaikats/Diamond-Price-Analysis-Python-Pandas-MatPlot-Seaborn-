import seaborn as sns

df = sns.load_dataset("diamonds")

# save to CSV
df.to_csv("diamonds.csv", index=False)

print("CSV file created successfully!")
