import pandas as pd

# Specify the path to your CSV file
csv_file_path = 'C:/Users/luigi/Desktop/Third Year/Business Intelligence/CIS3087---Business-Intelligence-Assignment/Facts/facts_seed_10.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter the DataFrame to include only 'Epoch' and 'Status' columns
filtered_df = df[['Epoch', 'Status']]

# Count the number of 'Bad Fact' entries for each epoch
bad_facts_count = filtered_df[filtered_df['Status'] == 'Bad Fact'].groupby('Epoch').size().reset_index(name='No of Bad Facts')

# Specify the path to save the results CSV file
output_csv_path = 'C:/Users/luigi/Desktop/Third Year/Business Intelligence/CIS3087---Business-Intelligence-Assignment/bad_facts_results.csv'

# Save the results to a new CSV file
bad_facts_count.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
