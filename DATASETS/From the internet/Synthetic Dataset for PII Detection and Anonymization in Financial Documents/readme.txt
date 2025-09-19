This dataset is designed for Training and Testing Machine Learning Models for Detecting and Anonymizing Personally Identifiable Information (PII) in financial documents. This dataset adheres to the highest data privacy standards and is fully synthetic, ensuring no real-world personal data is included. The dataset simulates various PII entities typically found in financial contexts. It is intended to support the development and evaluation of PII Detection and Anonymization Models. It includes training and testing sets of Synthetic Entries generated using realistic financial document structures.

Each entry simulates real-world financial texts such as auditor reports, tax filings, compliance notices, and transaction confirmations. The dataset contains a variety of Synthetic PII types embedded into these documents, including:

•	Names
•	Social Security Numbers (SSNs)
•	Credit Card Numbers
•	Phone Numbers
•	Email Addresses
•	Physical Addresses
•	Company Names
•	URLs

Dataset Structure:

The Training and Testing Datasets have the following structure of Synthetic Data:

Columns:

1.	Name: Contains the synthetic full names of individuals, generated with a mix of genders and cultural backgrounds.

2.	Credit Card: Lists synthetic credit card information, including card numbers, expiration dates, and security codes. Various credit card types (e.g., VISA, MasterCard, American Express) are represented.

3.	Email: Includes synthetic email addresses in realistic formats with diverse domain names.

4.	URL: Contains synthetic website URLs from various domains (e.g., .com, .org, .info), mimicking the variety found in real financial documents.

5.	Phone: Represents synthetic phone numbers in different formats, including international formats.

6.	Address: Consists of detailed synthetic addresses, including street names, cities, states, and postal codes, generated in various formats.

7.	Company: Includes synthetic company names across different industries, providing a realistic mix of common and unique names.

8.	SSN: Synthetic Social Security Numbers (SSNs) presented in various formats, including different region-specific patterns (e.g., with or without hyphens).

9.	Text: The main body of text simulating financial document content such as audits, reports, invoices, compliance notices, or transaction confirmations. Each text entry contains embedded PII data.

10.	True Predictions: Lists and Annotates the exact starting and ending character positions of each PII entity within the "Text" column, along with the entity type (e.g., 'name', 'email', 'address', etc.).

Please Note: This dataset does not contain any real-world Sensitive Information.