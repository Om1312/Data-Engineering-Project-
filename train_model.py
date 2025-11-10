import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import sklearn
import pickle
import warnings
import time
import gzip

# Optional: Hide less important warnings
warnings.filterwarnings("ignore")

print("🔹 Starting BigMart Model Training...")

# === 1. Connect to MySQL and Load Data ===
try:
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Om@04007',   # ✅ Your MySQL password
        database='BigMart'
    )

    df_item = pd.read_sql("SELECT * FROM item_info", connection)
    df_outlet = pd.read_sql("SELECT * FROM outlet_info", connection)
    df_sales = pd.read_sql("SELECT * FROM sales_info", connection)
    connection.close()

    print("✅ MySQL connection successful and data loaded.")
except Exception as e:
    print(f"❌ Database connection or loading failed: {e}")
    exit()

# === 2. Merge DataFrames ===
try:
    df = df_item.merge(df_outlet, on='ID').merge(df_sales, on='ID')
    df.drop('ID', axis=1, inplace=True)
    print(f"✅ Data merged successfully. Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error merging data: {e}")
    exit()

# === 3. Feature Engineering ===
try:
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df.drop('Outlet_Establishment_Year', axis=1, inplace=True)

    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    })

    # Cap extreme visibility values
    df['Item_Visibility'] = np.where(df['Item_Visibility'] > 0.3, 0.3, df['Item_Visibility'])

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    print("✅ Feature engineering and missing value handling complete.")
except Exception as e:
    print(f"❌ Error during feature engineering: {e}")
    exit()

# === 4. Prepare X, y ===
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# === 5. Identify Categorical and Numerical Columns ===
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# === 6. Preprocessing Pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# === 7. Define Models (Optimized for Speed) ===
models = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,      # Reduced from 200 to 100 for faster training
        learning_rate=0.05,    # Smaller learning rate (balanced)
        max_depth=3,           # Restrict tree depth
        subsample=0.7,         # Use 70% data for each iteration
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1              # Parallel processing
    ),
    "LinearRegression": LinearRegression()
}

# === 8. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 9. Train and Evaluate Models ===
best_model_name = None
best_score = -np.inf
best_pipeline = None
results = []

print("\n🚀 Training and Evaluating Models...")

for name, reg in models.items():
    print(f"\n🔹 Training {name}...")
    try:
        start_time = time.time()

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ Works in all sklearn versions

        print(f"📊 {name} Results:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:,.2f}")
        print(f"   ⏱️ Training Time: {train_time:.2f} seconds")

        results.append({
            'Model': name,
            'R² Score': r2,
            'RMSE': rmse,
            'Time (s)': round(train_time, 2)
        })

        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_pipeline = pipeline

    except Exception as e:
        print(f"❌ Error while training {name}: {e}")

# === 10. Display Model Summary ===
results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
print("\n📋 Model Performance Summary:")
print(results_df.to_string(index=False))

# === 11. Save Best Model ===
try:
    model_artifacts = {
        'pipeline': best_pipeline,
        'sklearn_version': sklearn.__version__,
        'columns': X.columns.tolist()
    }

    import gzip

    with gzip.open("bigmart_best_model.pkl.gz", "wb") as f:
        pickle.dump(model_artifacts, f)


    print(f"\n✅ Best Model: {best_model_name} (R² = {best_score:.4f}) saved successfully as bigmart_best_model.pkl")

except Exception as e:
    print(f"❌ Error saving model: {e}")
