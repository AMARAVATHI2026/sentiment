import pandas as pd
import nltk
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/reviews.csv")

print("Dataset Loaded Successfully\n")
print(df.head())