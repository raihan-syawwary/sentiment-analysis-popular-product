#!/usr/bin/env python
# coding: utf-8

# # Amazon Analyzing Product Reviews for Sentiment Analysis and Popular Products Identification

# ### Project Overview 
# 
# In this project, I'm going to analyze the product reviews from your Amazon e-commerce dataset to perform sentiment analysis and popular products. By leveraging the reviews and other relevant information, I will gain insights into customer sentiments and popular products
# 
# #### Note: 
# 
# This analysis is specific to Computer & Accessories items only 

# ### Data Overviewing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


data_path = os.path.join("amazon.csv")
print(data_path)


# In[3]:


data = pd.read_csv(data_path)


# In[4]:


data.columns


# In[5]:


data.head()


# In[6]:


data.isna().mean() * 100


# In[7]:


data.info()


# ### Data Preprocessing

# In[8]:


data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['discount_percentage'] = data['discount_percentage'].str.replace('%', '').astype(float)
data['rating'] = data['rating'].replace('|', np.nan).astype(float)
data['rating_count'] = data['rating_count'].str.replace(',','').astype(float)
data.info()


# In[9]:


data['rating'] = data['rating'].fillna(data['rating'].mean())
data['rating_count'] = data['rating_count'].fillna(data['rating_count'].mean())
data.info()


# In[10]:


data['category'] = data['category'].fillna('') 

data['sub_category_1'] = data['category'].apply(lambda x: x.split('|')[0] if len(x.split('|')) > 0 else '')
data['sub_category_2'] = data['category'].apply(lambda x: x.split('|')[1] if len(x.split('|')) > 1 else '')
data['sub_category_3'] = data['category'].apply(lambda x: x.split('|')[2] if len(x.split('|')) > 2 else '')
data['sub_category_4'] = data['category'].apply(lambda x: x.split('|')[3] if len(x.split('|')) > 3 else '')
data['sub_category_5'] = data['category'].apply(lambda x: x.split('|')[4] if len(x.split('|')) > 4 else '')

data.head()


# In[11]:


data.info()


# In[12]:


data = data.drop('category', axis=1)


# In[13]:


data.info()


# In[14]:


data = data[(data['sub_category_1'] == 'Computers&Accessories') & (data['sub_category_2'] == 'Accessories&Peripherals')]
data.info()


# In[15]:


data['rating_count'] = data['rating_count'].astype(int)
data.info()


# ### Sentiment Analysis

# In[16]:


import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# In[17]:


sentiment_sorted = data.sort_values(by='rating_count', ascending=False)
sentiment_filtered = sentiment_sorted.drop_duplicates(subset='product_id', keep='first')
data = sentiment_filtered
data.head()


# In[18]:


review_content = "Good product,long wire,Charges good,Nice,I bought this cable for Rs.339 worthy product for this price, i tested it in various charger adapters 33w and 18w it supports fast charging as well.,Good,Ok,I had got this at good price on sale on Amazon and product is useful with warranty but for warranty you need to go very far not practical for such a cost and mine micro to type c connector stopped working after few days only.,I like this product"
filtered_data = data[data['review_content'].str.contains(review_content)]
for index, row in filtered_data.iterrows():
    print("Product ID:", row['product_id'])
    print("Review Content:", row['review_content'])
    print()


# In[19]:


columns_to_drop = ['product_id', 'product_name', 'discounted_price', 'actual_price', 'discount_percentage',
                   'rating_count', 'about_product', 'user_id', 'user_name', 'review_id', 'review_title', 'img_link',
                   'product_link', 'sub_category_1', 'sub_category_2', 'sub_category_3', 'sub_category_4', 'sub_category_5']
data_sentiment = data.drop(columns=columns_to_drop)


# In[20]:


data_sentiment.info()


# In[21]:


nltk.download('stopwords')
nltk.download('punkt')

stopwords_set = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords_set and token not in punctuation_set]
    processed_text = ''.join(tokens)
    return processed_text

data_sentiment['review_content_processed'] = data_sentiment['review_content'].apply(preprocess_text)


# In[22]:


data_sentiment.head()


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

X = data_sentiment['review_content']
y = data_sentiment['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


# In[25]:


threshold = 3.5

y_train_binary = y_train >= threshold
y_test_binary = y_test >= threshold

classifier = LogisticRegression()
classifier.fit(X_train_bow, y_train_binary)


# In[26]:


#Accuracy 
y_pred = classifier.predict(X_test_bow)
accuracy = accuracy_score(y_test_binary, y_pred)
print("Accuracy:", accuracy)


# In[27]:


results = pd.DataFrame({'Actual Sentiment' : y_test_binary, 'Predicted Sentiment': y_pred})
print(results)


# In[28]:


total_samples = len(y_pred)
positive_count = sum(y_pred == 1)
negative_count = sum(y_pred == 0)
positive_proportion = (positive_count / total_samples) * 100 
negative_proportion = (negative_count / total_samples) * 100 

print("Proportion of Positive Sentiment: {:.2f}%". format(positive_proportion))
print("Proportion of Negative Sentiment: {:.2f}%". format(negative_proportion))


# In[29]:


y_actual = results['Actual Sentiment']
actual_total_samples = len(y_actual)
actual_positive_count = sum(y_pred == 1)
actual_negative_count = sum(y_pred == 0)
actual_positive_proportion = (positive_count / actual_total_samples) * 100 
actual_negative_proportion = (negative_count / total_samples) * 100 

print("Proportion of Actual Positive Sentiment: {:.2f}%". format(actual_positive_proportion))
print("Proportion of Actual Negative Sentiment: {:.2f}%". format(actual_negative_proportion))


# In[30]:


#Precision and Recall 
from sklearn.metrics import precision_score, recall_score
prec = precision_score(y_test_binary, y_pred)
recall = recall_score(y_test_binary, y_pred)
print("Precision:", prec, "Recall:", recall)


# In[31]:


#F1 Score
from sklearn.metrics import f1_score
f1 = f1_score(y_test_binary, y_pred)
print("F1 Score:", f1)


# In[32]:


from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[33]:


sia = SentimentIntensityAnalyzer()
def get_sentiment_scores(text): 
    scores = sia.polarity_scores(text)
    return scores['compound']

data_sentiment['review_content_analyzed'] = data_sentiment['review_content'].apply(get_sentiment_scores)


# In[34]:


data_sentiment.head()


# In[35]:


def categorize_sentiment(score, neutral_threshold=0.05): 
    if score > neutral_threshold:
        return 'positive'
    elif score < -neutral_threshold:
        return 'negative'
    else: 
        return 'neutral'

data_sentiment['review_content_categorized'] = data_sentiment['review_content_analyzed'].apply(categorize_sentiment)


# In[36]:


data_sentiment.head()


# In[37]:


data_sentiment['review_content_categorized'].unique()


# In[38]:


sentiment_counts = data_sentiment['review_content_categorized'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.title('Sentiment Category Distribution')

plt.show()


# In[39]:


plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Category Distribution')
plt.show()


# ### Summary of Sentiment Analysis
# 
# #### Conclusion: 
# 
# I employed two distinct methods for sentiment analysis. The overall findings indicate that Amazon has garnered a predominantly positive impression from customers, surpassing the negative sentiment. While there is a minor presence of neutral sentiment, the impact is on lesser significance of overall results. 
# 
# ###### 1. Logistic Regression Classifier: 
# The model demonstrates a high accuracy rate of 93.54% in accurately predicting both positive and negative sentiments. Furthermore, the review analysis for Accessories and Peripherals products within the Computer & Accessories category overall has garnered a 98.4%, both actual and predicted, positive impression from customers. 
# 
# ###### 2. Sentiment Intensity Analyzer: 
# 
# The employed sentiment analysis methodology has categorized the scores of the review_content column into three distinct categories: 'Positive', 'Neutral', and 'Negative'. Following the conducted analysis, it has been observed that positive impressions account for 94.5% of the chart's composition. These results closely align with the findings of the logistic regression classifier, indicating a consistent outcome.
# 
# 
# #### Recommendation 
# 
# The company must prioritize the maintenance of its performance and service levels for customers, while continuously striving to enhance them further. This commitment aims to provide customers with the utmost satisfaction and ensure their positive experience and contented departure.

# ### Product Popularity Analysis 

# In[40]:


popular_by_rating_count = data.groupby('product_id')['rating_count'].sum()
print(popular_by_rating_count)


# In[41]:


sorted_popular_by_rating_count = popular_by_rating_count.sort_values(ascending=False)
N = 10
top_product_rating_count = sorted_popular_by_rating_count.head(N)

plt.bar(top_product_rating_count.index, top_product_rating_count.values)
plt.xlabel('Product ID')
plt.ylabel('Rating Count')
plt.title('Top {} Popular Products Based on Rating Count'.format(N))
plt.xticks(rotation=90)
plt.show()


# In[42]:


top_product_rating_count.head(N)


# In[43]:


top_product_rating_count = top_product_rating_count.to_frame().reset_index()
top_10_products_with_names = top_product_rating_count.merge(data[['product_id', 'product_name']], on='product_id')
top_10_products_with_names[['product_id', 'product_name', 'rating_count']].head(N)


# In[44]:


product_rating = data.sort_values(by='rating', ascending=False)
product_rating.head()


# In[45]:


N = 10
top_products = product_rating.head(N)
top_products.head()


# In[46]:


top_10_products = top_products.groupby('product_id')['rating'].apply(list)
print(top_10_products)


# In[47]:


sorted_top_10_products = top_10_products.sort_values(ascending=False)

plt.bar(sorted_top_10_products.index, sorted_top_10_products.values)
plt.xlabel('Product ID')
plt.ylabel('Rating')
plt.title('Top {} Popular Products Based on Rating'.format(N))
plt.xticks(rotation=90)
plt.show()


# In[48]:


top_10_products = sorted_top_10_products.to_frame().reset_index()
top_10_names = top_10_products.merge(data[['product_id', 'product_name']], on='product_id')
top_10_names[['product_id', 'product_name', 'rating']].head(N)


# ### Summary of Popular Product Analysis 
# 
# These two analyses were carried out independently, with one focusing on the total number of reviews (rating_count), while the other centered around rating scores (rating).
# 
# #### 1. Analysis Based on Total Number of Reviews 
# 
# According to the total number of reviews, the following products emerge as the most popular: 
# 
# 1. Amazon Basics USB A to Lightning MFi Certified Charging Cable (White, 1.2 meter)	
# 
# 2. AmazonBasics USB 2.0 Cable - A-Male to B-Male - for Personal Computer, Printer- 6 Feet (1.8 Meters), Black	
# 
# 3. boAt Rugged v3 Extra Tough Unbreakable Braided Micro USB Cable 1.5 Meter (Black)	
# 
# 4. boAt Deuce USB 300 2 in 1 Type-C & Micro USB Stress Resistant, Tangle-Free, Sturdy Cable with 3A Fast Charging & 480mbps Data Transmission, 10000+ Bends Lifespan and Extended 1.5m Length(Martian Red)
# 
# 5. boAt Rugged V3 Braided Micro USB Cable (Pearl White)
# 
# 6. boAt Deuce USB 300 2 in 1 Type-C & Micro USB Stress Resistant, Sturdy Cable with 3A Fast Charging & 480mbps Data Transmission, 10000+ Bends Lifespan and Extended 1.5m Length(Mercurial Black)
# 
# 7. AmazonBasics Micro USB Fast Charging Cable for Android Phones with Gold Plated Connectors (3 Feet, Black)	
# 
# 8. Amazonbasics Micro Usb Fast Charging Cable For Android Smartphone,Personal Computer,Printer With Gold Plated Connectors (6 Feet, Black)	
# 
# 9. AmazonBasics USB 2.0 Extension Cable for Personal Computer, Printer, 2-Pack - A-Male to A-Female - 3.3 Feet (1 Meter, Black)
# 
# 10. AmazonBasics USB 2.0 - A-Male to A-Female Extension Cable for Personal Computer, Printer (Black, 9.8 Feet/3 Meters)	
# 
# #### 2. Analysis Based on Highest Rating 
# 
# Based on the ratings received, the following ten products stand out with the highest ratings: 
# 
# 1. Amazon Basics Wireless Mouse | 2.4 GHz Connection, 1600 DPI | Type - C Adapter | Upto 12 Months of Battery Life | Ambidextrous Design | Suitable for PC/Mac/Laptop
# 
# 2. Syncwire LTG to USB Cable for Fast Charging Compatible with Phone 5/ 5C/ 5S/ 6/ 6S/ 7/8/ X/XR/XS Max/ 11/12/ 13 Series and Pad Air/Mini, Pod & Other Devices (1.1 Meter, White)
# 
# 3. REDTECH USB-C to Lightning Cable 3.3FT, [Apple MFi Certified] Lightning to Type C Fast Charging Cord Compatible with iPhone 14/13/13 pro/Max/12/11/X/XS/XR/8, Supports Power Delivery - White
# 
# 4. Logitech G402 Hyperion Fury USB Wired Gaming Mouse, 4,000 DPI, Lightweight, 8 Programmable Buttons, Compatible for PC/Mac - Black
# 
# 5. Redgear MP35 Speed-Type Gaming Mousepad (Black/Red)
# 
# 6. Logitech M331 Silent Plus Wireless Mouse, 2.4GHz with USB Nano Receiver, 1000 DPI Optical Tracking, 3 Buttons, 24 Month Life Battery, PC/Mac/Laptop - Black	
# 
# 7. Logitech Pebble M350 Wireless Mouse with Bluetooth or USB - Silent, Slim Computer Mouse with Quiet Click for Laptop, Notebook, PC and Mac - Graphite	[4.6]
# 
# 8. AmazonBasics New Release Nylon USB-A to Lightning Cable Cord, MFi Certified Charger for Apple iPhone, iPad, Silver, 6-Ft	
# 
# 9. SupCares Laptop Stand 7 Height Adjustable, Aluminium, Ventilated, Foldable, Portable Laptop Holder for Desk & Table Mount Upto 15.6 inch Laptop with Carry Pouch (Silver)	
# 
# 10. Lapster 65W compatible for OnePlus Dash Warp Charge Cable , type c to c cable fast charging Data Sync Cable Compatible with One Plus 10R / 9RT/ 9 pro/ 9R/ 8T/ 9/ Nord & for All Type C Devices – Red, 1 Meter
