import pandas as pd

file = pd.read_csv('my_dataset.csv')
file2 = pd.read_excel('dataset.xlsx')

cols = file.columns
cols2 = file2.columns

cols = [i.strip() for i in cols]
cols2 = [i.strip() for i in cols2]

# for i in cols2:
#     if i not in cols:
#         print(i)
"""
(base) friday_code@gaurav-kushwaha:~/Desktop/DP-Regenrate-Results$ python main.py 
ddGglyc
ddGglyc error
"""

# add above two missing columns in file in a new dataframe and save it as csv
missing_cols = [i for i in cols2 if i not in cols]
print(missing_cols)

# match name and interactor columns in both files and add the missing columns to file
file2 = file2.set_index(['name', 'interactor'])
file = file.set_index(['name', 'interactor'])
for col in missing_cols:
    file[col] = file2[col]

file.to_csv('my_dataset.csv')