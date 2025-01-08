from tabulate import tabulate

def display_df(dataframe):
    print(tabulate(dataframe.head(), headers='keys', tablefmt='psql'))
    print("Dataset Overview:")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print("\nColumn Types:")
    print(dataframe.dtypes)
    print("\nMissing Values:")
    print(dataframe.isnull().sum())
    print("\nDataframe Info:")
    print(dataframe.info)