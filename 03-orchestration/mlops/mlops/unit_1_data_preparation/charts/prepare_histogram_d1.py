import pandas as pd

from mage_ai.shared.parsers import convert_matrix_to_dataframe


if isinstance(df_1, list) and len(df_1) >= 1:
    item = df_1[0]
    if isinstance(item, pd.Series):
        item = item.to_frame()
    elif not isinstance(item, pd.DataFrame):
        item = convert_matrix_to_dataframe(item)
    df_1 = item

columns = df_1.columns
# col = list(filter(lambda x: df_1[x].dtype == float or df_1[x].dtype == int, columns))[0]
col = 'trip_distance'
x = df_1[df_1[col] <= 20][col]
