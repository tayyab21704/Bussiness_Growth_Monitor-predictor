
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


data = pd.read_csv(r"C:\fds_BusinessModel\supermarket_sales - Sheet1.csv")  # Using a raw string literal for the file path


data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month


label_encoders = {}
categorical_columns = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


features = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Day_of_Week', 'Month', 'Quantity', 'Tax 5%', 'gross income']
target = 'Total'


X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)


model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


product_lines = data['Product line'].unique()
product_sales = []
for product in product_lines:
    product_data = data[data['Product line'] == product]
    prediction = model.predict(product_data[features].mean().to_frame().T)
    product_sales.append((product, prediction[0]))


product_sales = sorted(product_sales, key=lambda x: x[1], reverse=True)
product_names, predicted_sales = zip(*product_sales)


inverse_label_encoders = {col: le.inverse_transform(data[col]) for col, le in label_encoders.items()}
data.replace(inverse_label_encoders, inplace=True)


plt.figure(figsize=(10, 6))
sns.barplot(x='Product line', y='Predicted Sales', hue='Product line', data=pd.DataFrame({'Product line': product_names, 'Predicted Sales': predicted_sales}), palette='viridis', dodge=False)
plt.xlabel('Product Line')
plt.ylabel('Predicted Sales')
plt.title('Predicted Sales by Product Line')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Customer type', hue='Gender', data=data, palette='pastel')
plt.xlabel('Customer Type')
plt.ylabel('Count')
plt.title('Customer Type by Gender')
plt.legend(title='Gender', loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='City', hue='Product line', data=data, palette='muted')
plt.xlabel('City')
plt.ylabel('Number of Sales')
plt.title('City-wise Sale of Product Line')
plt.legend(title='Product Line', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


investment_recommendations = {product: sales for product, sales in product_sales}
print("Investment Recommendations (Product Line and Predicted Sales):")
for product, sales in investment_recommendations.items():
    print(f"{product}: {sales:.2f}")
