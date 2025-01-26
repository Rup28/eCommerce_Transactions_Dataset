import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")
products = pd.read_csv("Products.csv")

# Merge datasets
data = transactions.merge(customers, on="CustomerID", how="left")
data = data.merge(products, on="ProductID", how="left")

#__________Region wise Segregation of high valued customer_________

# Total revenue per customer
customer_revenue = data.groupby(['CustomerID', 'Region']).agg({'TotalValue': 'sum'}).reset_index()
customer_revenue.rename(columns={'TotalValue': 'TotalRevenue'}, inplace=True)

# Top 20% customers
top_20_percent = customer_revenue['TotalRevenue'].quantile(0.8)
high_value_customers = customer_revenue[customer_revenue['TotalRevenue'] >= top_20_percent]
print(high_value_customers.head(20))
# Visualization
sns.countplot(data=high_value_customers, x='Region',palette='viridis')
plt.title("High-Value Customers by Region")
plt.xlabel("Region")
plt.ylabel("Number of High-Value Customers")
plt.show()


#__________Product Category Revenue Contribution____________

# Total revenue by product
product_sales = data.groupby(['ProductName', 'Category'])['TotalValue'].sum().sort_values(ascending=False).reset_index()

# Top 5 best-selling products
top_products = product_sales.head(5)
print("Top 5 Best-Selling Products:\n", top_products)

# Revenue by category
category_sales = data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()

#__________Monthly Revenue Seasonality trend______________

# Convert TransactionDate to datetime
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

# Group by month
data['Month'] = data['TransactionDate'].dt.month
monthly_sales = data.groupby('Month')['TotalValue'].sum()

# Plot monthly sales
monthly_sales.plot(kind='bar', color='orange', figsize=(10, 5))
plt.title("Monthly Revenue (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()



#__________Customer Transaction Frequency Distribution_________


# Transactions per customer
repeat_customers = data.groupby('CustomerID')['TransactionID'].count()

# Percentage of repeat customers
repeat_percentage = (repeat_customers[repeat_customers > 1].count() / repeat_customers.count()) * 100
print(f"Percentage of Repeat Customers: {repeat_percentage:.2f}%")

# Plot repeat purchase distribution
sns.histplot(repeat_customers, bins=10, kde=False, color='purple')
plt.title("Distribution of Customer Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Number of Customers")
plt.show()

#_________Low performing products_________________
# Revenue by product
product_revenue = data.groupby('ProductName')['TotalValue'].sum().reset_index()

# Bottom 20% products
low_performing_products = product_revenue[product_revenue['TotalValue'] <= product_revenue['TotalValue'].quantile(0.2)]

# Plot low-performing products
low_performing_products.sort_values(by='TotalValue', ascending=True).plot(
    x='ProductName', y='TotalValue', kind='barh', color='red', figsize=(8, 6)
)
plt.title("Low-Performing Products")
plt.xlabel("Revenue")
plt.ylabel("Product Name")
plt.show()

print("Low-Performing Products:\n", low_performing_products)
