# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\uchen\PycharmProjects\Medicines\Medicine_Details.csv")

# Data Overview
print(data.head(4))

# Data information
print(data.info())

# Statistical summary
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Histplot visualization of excellent reviews with number of medicine
plt.figure(figsize=(15, 6))

# Plotting distributions of reviews
plt.subplot(1, 3, 1)
sns.histplot(data['Excellent Review %'], bins=30, kde=True, color='green')
plt.title('Distribution of Excellent Reviews %')
plt.xlabel('Excellent Review %')
plt.ylabel('Number of Medicines')

plt.tight_layout()
print(plt.show())

# Histplot visualization of average reviews with number of medicines
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 2)
sns.histplot(data['Average Review %'], bins=30, kde=True, color='blue')
plt.title('Distribution of Average Reviews %')
plt.xlabel('Average Review %')
plt.ylabel('Number of Medicines')

plt.tight_layout()
print(plt.show())

# Histplot visualization of poor reviews with number of Medicines
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 3)
sns.histplot(data['Poor Review %'], bins=30, kde=True, color='red')
plt.title('Distribution of Poor Reviews %')
plt.xlabel('Poor Review %')
plt.ylabel('Number of Medicines')

plt.tight_layout()
print(plt.show())

# Top 10 manufacturers by number of medicines
top_manufacturers = data['Manufacturer'].value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(y=top_manufacturers.index, x=top_manufacturers.values, palette="viridis")
plt.title('Top 10 Manufacturers by Number of Medicines')
plt.xlabel('Number of Medicines')
plt.ylabel('Manufacturer')

print(plt.show())

# Top 10 most common compositions
top_compositions = data['Composition'].value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(y=top_compositions.index, x=top_compositions.values, palette="magma")
plt.title('Top 10 Most Common Compositions')
plt.xlabel('Number of Medicines')
plt.ylabel('Composition')

print(plt.show())

# Splitting the uses and counting the occurrences
uses = data['Uses'].str.split(',').explode().str.strip().value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(y=uses.index, x=uses.values, palette="cubehelix")
plt.title('Top 10 Most Common Uses of Medicines')
plt.xlabel('Number of Medicines')
plt.ylabel('Use')

print(plt.show())

# Splitting the side effects and counting the occurrences
side_effects = data['Side_effects'].str.split(',').explode().str.strip().value_counts().head(10)

plt.figure(figsize=(12, 7))
sns.barplot(y=side_effects.index, x=side_effects.values, palette="coolwarm")
plt.title('Top 10 Most Common Side Effects of Medicines')
plt.xlabel('Number of Medicines')
plt.ylabel('Side Effect')

print(plt.show())

# Selecting top manufacturers by number of medicines they produce
top_manufacturers_list = top_manufacturers.index.tolist()

print(top_manufacturers_list)

# Filtering data for these manufacturers and calculating their average review ratings
manufacturer_avg_reviews = data[data['Manufacturer'].isin(top_manufacturers_list)].groupby('Manufacturer')[['Excellent Review %', 'Average Review %', 'Poor Review %']].mean()

print(manufacturer_avg_reviews)

# Sorting manufacturers by average of Excellent Review %
manufacturer_avg_reviews_sorted = manufacturer_avg_reviews.sort_values(by='Excellent Review %',ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(y=manufacturer_avg_reviews_sorted.index, x=manufacturer_avg_reviews_sorted['Excellent Review %'], palette="Blues_d")
plt.title('Average Excellent Review % of Top Manufacturers')
plt.xlabel('Average Excellent Review %')
plt.ylabel('Manufacturer')

print(plt.show())

# To visualize the spread between excellent, average, and poor reviews
plt.figure(figsize=(15, 6))

# Scatter plot to visualize the spread
plt.scatter(data['Excellent Review %'], data['Average Review %'], color='green', label='Average Review %', alpha=0.5)
plt.scatter(data['Excellent Review %'], data['Poor Review %'], color='red', label='Poor Review %', alpha=0.5)

plt.title('Spread of Reviews')
plt.xlabel('Excellent Review %')
plt.ylabel('Review %')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

print(plt.show())

# Weighted Review score
data['review_score'] = (
    (data['Excellent Review %'] * 5 + data['Average Review %'] * 3 + data['Poor Review %'] * 1) /
    (data['Excellent Review %'] + data['Average Review %'] + data['Poor Review %'] + 1e-10)

)

print(data.head(3))

# Group by key factors
effectiveness_by_use = data.groupby('Uses')['review_score'].mean().sort_values(ascending=False)
effectiveness_by_manufacturer = data.groupby('Manufacturer')['review_score'].mean().sort_values(ascending=False)
effectiveness_by_salt = data.groupby('Composition')['review_score'].mean().sort_values(ascending=False)

# Visualize top 10 effectiveness by use
effectiveness_by_use.head(10).plot(kind='bar', title='Top 10 Effectiveness by Use', figsize=(10, 5))

print(plt.show())

# Visualize top 10 effectiveness by manufacturers
effectiveness_by_manufacturer.head(10).plot(kind='bar', title='Top 10 Effectiveness by Manufacturer', figsize=(10, 5))

print(plt.show())

# Visualize the top 10 effectiveness by salt
effectiveness_by_salt.head(10).plot(kind='bar', title='Top 10 Effectiveness by Salt Composition', figsize=(10, 5))

print(plt.show())

# To create a word cloud for side effects to get the most common side effects of medicines
from wordcloud import WordCloud

side_effects_text = ' '.join(data['Side_effects'])
wordcloud_side_effects = WordCloud(background_color='white', width=800, height=400, max_words=100).generate(side_effects_text)

# Plotting the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_side_effects, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Side Effects of Medicines')

print(plt.show())

# Tokenize side effects column and count the frequency
from collections import Counter
from nltk.tokenize import word_tokenize
# Download the 'punkt_tab' data for tokenization
import nltk
nltk.download('punkt_tab')

# Tokenize side effects
data['side_effects_tokens'] = data['Side_effects'].apply(lambda x: word_tokenize(x.lower()))

# Count frequency of side effects
side_effects_counter = Counter([effect for effects in data['side_effects_tokens'] for effect in effects])
most_common_side_effects = pd.DataFrame(side_effects_counter.most_common(20), columns=['Side_effects', 'count'])

# Plot
most_common_side_effects.plot(kind='bar', x='Side_effects', y='count', title='Most Common Side Effects', figsize=(10, 5))

print(plt.show)

# Identify which medicines are associated with the most common side effects
def find_medicines_for_side_effect(effect, data):
    return data[data['Side_effects'].str.contains(effect, case=False)][['Medicine Name', 'Side_effects']]

# To find the medicines and their side effects
side_effects_medicines = find_medicines_for_side_effect('headache', data)

print(side_effects_medicines)

# Tokenize side effects
data['side_effects_tokens'] = data['Side_effects'].apply(lambda x: word_tokenize(x.lower()))

# Flatten the list of side effects to count frequency
all_side_effects = [effect for effects in data['side_effects_tokens'] for effect in effects]

# Count the frequency of each side effects
side_effects_counter = Counter(all_side_effects)

# Display the most common side effects
most_common_side_effects = pd.DataFrame(side_effects_counter.most_common(20), columns=['Side_effects', 'Count'])

print('Most Common Side Effects:')
print(most_common_side_effects)

# Group medicines by side effects
side_effects_to_medicines = {}

for idx, row in data.iterrows():
  for side_effect in row['side_effects_tokens']:
    if side_effect not in side_effects_to_medicines:
      side_effects_to_medicines[side_effect] = []
      side_effects_to_medicines[side_effect].append(row['Medicine Name'])

# Display a sample of side effects and their associated medicines
sample_side_effects = list(side_effects_to_medicines.keys())[:10]
for side_effect in sample_side_effects:
  print(f'Side Effect: {side_effect}')
  print(f'Associated Medicines: {side_effects_to_medicines[side_effect]}')
  print('-' * 40)

# Plot the most common side effects
plt.figure(figsize=(10, 6))
plt.barh(most_common_side_effects['Side_effects'], most_common_side_effects['Count'])
plt.xlabel('Frequency')
plt.ylabel('Side_effects')
plt.title('Most Common Side Effects')
plt.gca().invert_yaxis()

print(plt.show())