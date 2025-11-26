import pymupdf
import camelot
import tabula
import pandas as pd 


# def check_pymudf(page_no, file_name):
#     doc = pymupdf.open(file_name)
#     page = doc.load_page(page_no - 1)  # 0-based index
#     tables = page.get_text("dict")["blocks"]
#     table_count = sum(1 for block in tables if block['type'] == 1)  # type 1 indicates a table
#     print(f"pymupdf found {table_count} tables on page {page_no}.")
#     print(tables)

def check_camelot(page_no, file_name):
    tables = camelot.read_pdf(file_name, pages=str(page_no))
    print(f"Camelot found {tables.n} tables on page {page_no}.")
    for i, table in enumerate(tables):
        print(f"Table {i + 1}:\n", table.df)
        # print the no od rows and columns
        print(f"Table {i + 1} has {table.df.shape[0]}")
    

# def check_tabula(page_no, file_name):
#     tables = tabula.read_pdf(file_name, pages=page_no, multiple_tables=True)
#     print(f"Tabula found {len(tables)} tables on page {page_no}.")
#     for i, table in enumerate(tables):
#         print(f"Table {i + 1}:\n", table)
if __name__ == "__main__":
    file_name = "2025-10-01_10-1615_昼_A.pdf"
    file_name2 ="2025-11-17_10-1615_昼_B.pdf"
    page_no = 2
    # check_pymudf(page_no, file_name)
    check_camelot(page_no, file_name)   
    # check_tabula(page_no, file_name)


