
## Adding new sources to an existing source file


```python
import pandas as pd
import numpy as np
```


```python
def add_new_data(old_data, new_data, old_labels, label):
    """
    Update the sourcepredict learning table
    INPUT:
        old_data(str): path to csv file of existing sourcepredict source data table
        new_data(str): path to csv file of new OTU table, with TAXID as 1st column
        old_labels(str): path to sourcepredict csv file of labels
        label(str): scientific name of new sample's specie. Example: 'Sus_scrofa'
    OUTPUT:
        merged(pd.DataFrame): merged old and new source data table for sourcepredict
        labels(pd.DataFrame): updated labels data table
    """
    old = pd.read_csv(old_data, index_col=0)
    old = old.drop(['labels'], axis = 0)
    new = pd.read_csv(new_data)
    merged = pd.merge(left=old, right=new, how='outer', on='TAXID')
    merged = merged.fillna(0)
    old_labels = pd.read_csv(old_labels, index_col=0)
    new_labels = pd.DataFrame([label]*(new.shape[1]-1), new.columns[1:])
    new_labels.columns=['labels']
    labels = old_labels.append(new_labels)
    return(merged, labels)
```


```python
labs = add_new_data(old_data=old_data, new_data=new_data, old_labels=old_labels, label=label)[1]
```


```python
labs.to_csv("new_sources.csv")
```
