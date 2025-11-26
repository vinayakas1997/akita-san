from pdf_csv import PDFTableExtractor

if __name__ == "__main__":
    extractor = PDFTableExtractor()
    file_path = "2025-10-01_10-1615_昼_A.pdf"
    results = extractor.process_clean_and_save(file_path, "check_output")
    extractor.print_summary()


    # folder_path = "test_folder"
    # results = extractor.process_clean_and_save(folder_path, "output2")
    # extractor.print_summary()