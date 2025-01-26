import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Configurable Parameters
TOP_N = 3  # Number of lookalikes to recommend
NUM_CUSTOMERS = 20  # First N customers to evaluate (e.g., C0001-C0020)

# Step 1: Load datasets
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")
products_df = pd.read_csv("Products.csv")

# Step 2: Map product categories to transactions
transactions_df = transactions_df.merge(products_df[['ProductID', 'Category']], on='ProductID', how='left')

# Step 3: Aggregate product interaction by customer and category
category_interaction = transactions_df.groupby(['CustomerID', 'Category']).agg({'Quantity': 'sum'}).reset_index()
category_pivot = category_interaction.pivot(index='CustomerID', columns='Category', values='Quantity').fillna(0)

# Step 4: Normalize product interaction data
scaler = MinMaxScaler()
category_pivot_scaled = pd.DataFrame(
    scaler.fit_transform(category_pivot), 
    columns=category_pivot.columns, 
    index=category_pivot.index
)

# Step 5: Merge customer data with region information
customers_df = customers_df.set_index('CustomerID')
final_df = customers_df.join(category_pivot_scaled, how='left').fillna(0)
final_df.drop(columns=["CustomerName","SignupDate"],inplace=True)

# Encode region and handle edge cases
region_columns = ['Region']
if all(col in final_df.columns for col in region_columns):
    encoder = OneHotEncoder(sparse_output=False)
    region_encoded = encoder.fit_transform(final_df[region_columns])
    region_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(region_columns), index=final_df.index)
    final_df = pd.concat([final_df.drop(columns=region_columns), region_df], axis=1)

# Step 6: Calculate cosine similarity
similarity_matrix = cosine_similarity(final_df)

# Step 7: Find top-N similar customers for each customer
similarity_df = pd.DataFrame(similarity_matrix, index=final_df.index, columns=final_df.index)
lookalike_map = {}

for customer_id in final_df.index[:NUM_CUSTOMERS]:  # First N customers
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:TOP_N + 1]  # Exclude self
    lookalike_map[customer_id] = [(sim_id, round(score, 4)) for sim_id, score in similar_customers.items()]

# Step 8: Save Lookalike Map to CSV
lookalike_df = pd.DataFrame({
    'CustomerID': lookalike_map.keys(),
    'Lookalikes': [str(v) for v in lookalike_map.values()]
})
lookalike_df.to_csv('Lookalike.csv', index=False)
