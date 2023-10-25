import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

dataset = []
with open('Market_Basket_Optimisation.csv') as file:
    reader = csv.reader(file,delimiter=',')
    for row in reader:
        dataset+=[row]

len(dataset)

te = TransactionEncoder()

x = te.fit_transform(dataset)
x

df = pd.DataFrame(x,columns=te.columns_)
df.head()

#find frequent items
freq_items = apriori(df,min_support=0.01,use_colnames=True)

freq_items

#now make the rules
rules = association_rules(freq_items,metric='confidence',min_threshold=0.25)
rules = rules[['antecedents','consequents','support','confidence']]
rules.head()
rules[rules['antecedents'] == {'cake'}]