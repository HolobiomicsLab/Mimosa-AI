import os
import pandas as pd

df = pd.read_csv('retraction_watch.csv')
df = df.drop(['Record ID','Institution','Journal','Publisher','Country','Author','ArticleType','RetractionDate','RetractionDOI','RetractionPubMedID','OriginalPaperDate','OriginalPaperDOI','OriginalPaperPubMedID','RetractionNature'], axis=1)
df.to_csv('papers.csv', index=False)

