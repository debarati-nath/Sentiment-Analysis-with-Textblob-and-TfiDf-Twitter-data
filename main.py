#Installation
!pip install textblob
!pip install scikit-learn
!pip install nltk
!pip install wordcloud

#After processing Tweets the csv file saved as below
tweets_raw["Processed"] = tweets_raw["Content"].str.lower().apply(process_tweets)
# Print the first fifteen rows of Processed
display(tweets_raw[["Processed"]].head(15))

#import libraries
import textblob
from textblob import TextBlob
# Add polarities and subjectivities into the DataFrame by using TextBlob
tweets_processed["Polarity"] = tweets_processed["Processed"].apply(lambda word:TextBlob(word).sentiment.polarity)
tweets_processed["Subjectivity"] = tweets_processed["Processed"].apply(lambda word:TextBlob(word).sentiment.subjectivity)

# Display the Polarity and Subjectivity columns
display(tweets_processed[["Polarity","Subjectivity"]].head(10))

# Define a function to classify polarities in tweets
def analyse_polarity(polarity):
    if polarity > 0:
        return "Positive"
    if polarity == 0:
        return "Neutral"
    if polarity < 0:
        return "Negative"

# Apply the function on Polarity column and add the results into a new column
tweets_processed["Polarity Scores"] = tweets_processed["Polarity"].apply(analyse_polarity)

# Display the Polarity and Subjectivity Analysis
display(tweets_processed[["Polarity Scores"]].head(10))

# Print the value counts of the Label column
print(tweets_processed[["Polarity Scores"]].value_counts())
#print(tweets_processed.rename(columns={'Label':'Polarity Scores'}, inplace=True))

#TfIDf
import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Import word_tokenize and stopwords from nltk
from nltk.corpus import stopwords
#Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words= stopwords.words('english'))
# Fit and transform the vectorizer
tfidf_matrix = vectorizer.fit_transform(tweets_processed[["Processed"]])
display(tfidf_matrix)
# Create a DataFrame for tf-idf vectors and display
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns= vectorizer.get_feature_names())
display(tfidf_df.head(5))

