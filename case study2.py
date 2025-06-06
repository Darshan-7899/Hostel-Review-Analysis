import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score    

#load dataset
df = pd.read_csv("hotel_reviews_large.csv")
print(df.head())
print(df.describe())
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())    



from textblob import TextBlob

# Perform sentiment analysis on 'review_text' column
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

# Assuming your review column is named 'review_text'
df['review_sentiment'] = df['Review_Text'].apply(get_sentiment)

# Check sentiment scores
print(df[['Review_Text', 'review_sentiment']].head())

plt.figure(figsize=(8,4))
sns.histplot(df['review_sentiment'], bins=30, kde=True, color='purple')
plt.title('Distribution of Review Sentiment')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Correlation between sentiment and review rating
correlation = df['review_sentiment'].corr(df['Review_Rating'])
print(f"Correlation between sentiment score and review rating: {correlation:.2f}")

# Scatter plot to visualize relationship
plt.figure(figsize=(8,4))
sns.scatterplot(x='Review_Rating', y='review_sentiment', data=df)
plt.title('Review Sentiment vs. Numeric Rating')
plt.xlabel('Review Rating')
plt.ylabel('Sentiment Score')
plt.tight_layout()
plt.show()
