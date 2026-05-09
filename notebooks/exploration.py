import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

os.makedirs('../results', exist_ok=True)

# Load data
train_df = pd.read_csv('../data/train.csv', header=None, names=['label', 'title', 'description'])
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
train_df['label'] = train_df['label'] - 1  # 0-indexed
train_df['class_name'] = train_df['label'].map(lambda x: class_names[x])
train_df['text'] = train_df['title'] + ' ' + train_df['description']

# Plot class distribution
plt.figure(figsize=(8, 5))
train_df['class_name'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Class Distribution in AG News')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/class_dist.png')

# Plot text lengths
train_df['text_len'] = train_df['text'].apply(lambda x: len(re.findall(r'\b\w+\b', x.lower())))
plt.figure(figsize=(10, 5))
plt.hist(train_df['text_len'], bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Text Lengths (in words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.axvline(train_df['text_len'].median(), color='red', linestyle='dashed', linewidth=1, label=f"Median: {train_df['text_len'].median():.0f}")
plt.legend()
plt.tight_layout()
plt.savefig('../results/text_len_dist.png')
