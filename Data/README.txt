The CSV files contain data evaluated on a mesh. The first row contains
row label in the first column, and then placeholder integers:

rowLabel, 0, 1, 2, 3, ...

The second row contains the column label followed by the column values:

colLabel, colVal1, colVal2, colVal3

For instance if the values are saved as a function of (x,t) with sample
data {(0,0):'A', (0.5,0):'B', (0,2.0):'C', (0.5,2.0):'D'}, this would be
stored as (with spaces added for clarity):

x,    0,   1
t,    0,   2.0
0,   'A', 'C'
0.5, 'B', 'D'
