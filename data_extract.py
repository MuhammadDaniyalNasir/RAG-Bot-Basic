# import gutenbergpy as gutenberg
# from gutenberg.acquire import load_etext
# from gutenberg.cleanup import strip_headers
# import os

# def download_gutenberg_books(book_ids, output_dir="gutenberg_books"):
#     """Download specific books by their Project Gutenberg IDs"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     for book_id in book_ids:
#         try:
#             # Download the book
#             text = strip_headers(load_etext(book_id)).strip()
            
#             # Save to file
#             filename = f"{output_dir}/book_{book_id}.txt"
#             with open(filename, 'w', encoding='utf-8') as f:
#                 f.write(text)
            
#             print(f"Downloaded book {book_id}")
#         except Exception as e:
#             print(f"Error downloading book {book_id}: {e}")

# # Example: Download some popular books
# popular_books = [11, 1342, 74, 84, 2701, 1661, 174, 345, 730, 1184]
# download_gutenberg_books(popular_books)

import pdfplumber

# with pdfplumber.open("/home/dani/lstm_project/nextword/lstm-next-word-predictor/derma.pdf") as pdf:
#     for i, page in enumerate(pdf.pages):
#         text = page.extract_text()
#         print(f"Page {i+1}")
#         print(text)

print (text)