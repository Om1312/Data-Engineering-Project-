import pymysql
import pandas as pd

connection = pymysql.connect(
    host = 'localhost',
    user = 'root',
    password = 'Om@04007',
    autocommit = True
)

cursor = connection.cursor()


# === 2. Create "BigMark" Database ===
cursor.execute("CREATE DATABASE IF NOT EXISTS BigMart")
cursor.execute("USE BigMart")

# === 3. Load the 3 XML Files ===
df_item = pd.read_xml('df_item.xml')
df_outlet = pd.read_xml('df_outlet.xml')
df_sales = pd.read_xml('df_sales.xml')

# === 4. Create Tables Automatically ===
def create_table_from_df(df, table_name):
    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if 'int' in str(dtype):
            sql_type = 'INT'
        elif 'float' in str(dtype):
            sql_type = 'FLOAT'
        else:
            sql_type = 'VARCHAR(255)'
        cols.append(f"`{col}` {sql_type}")
    
    columns_sql = ", ".join(cols)
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns_sql})"
    cursor.execute(create_sql)

# Create tables for each DataFrame
create_table_from_df(df_item, 'item_info')
create_table_from_df(df_outlet, 'outlet_info')
create_table_from_df(df_sales, 'sales_info')

# === 5. Insert Data into Tables (use executemany for speed) ===
def insert_data(df, table_name):
    df = df.where(pd.notnull(df), None)  # Replace NaN with None
    cols = ",".join([f"`{col}`" for col in df.columns])
    placeholders = ",".join(["%s"] * len(df.columns))
    insert_sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
    data = [tuple(row) for row in df.itertuples(index=False, name=None)]
    cursor.executemany(insert_sql, data)


# Insert data
insert_data(df_item, 'item_info')
insert_data(df_outlet, 'outlet_info')
insert_data(df_sales, 'sales_info')

# === 6. Done ===
print("✅ BigMart database created and all tables loaded successfully!")

# Close connection
cursor.close()
connection.close()