# This is a demo project..... 
# This project is about data collection(As feedbacks on a ecommerce website) and analysing that data furthermore.



import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('stopwords', quiet=True)


feedback_df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\Kartik's project\\customer_feedback_large.csv")  # Replace with your file path if needed


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


feedback_df['Cleaned Feedback'] = feedback_df['Feedback'].apply(clean_text)


feedback_df['Tokenized Feedback'] = feedback_df['Cleaned Feedback'].apply(lambda x: x.split())


feedback_df['Polarity'] = feedback_df['Cleaned Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)
feedback_df['Sentiment'] = feedback_df['Polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


feedback_df['Feedback Length'] = feedback_df['Feedback'].apply(len)


plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=feedback_df, palette='viridis')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


print(feedback_df[['Customer ID', 'Feedback', 'Cleaned Feedback', 'Sentiment', 'Polarity', 'Feedback Length']].head(10))


feedback_df.to_csv('processed_customer_feedback.csv', index=False)
