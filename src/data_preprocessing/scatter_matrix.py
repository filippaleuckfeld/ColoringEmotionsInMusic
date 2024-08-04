import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("src/feature_extraction/tracks_cleaned_features.csv")

features = ['energy', 'valence']

sns.pairplot(df[features])

plt.show()
