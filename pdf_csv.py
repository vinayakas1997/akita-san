import glob
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import camelot
import fitz  # PyMuPDF for page count AND text extraction
import pandas as pd
import re
from datetime import datetime, timedelta


class PDFTableExtractor:
    """
    Extract tables from PDF files with detailed file information parsing.
    Specializes in finding and extracting multi-page "その他実績" tables.
    """
    
    def __init__(self, flavor: str = 'lattice'):
        """Initialize the PDF table extractor."""
        self.flavor = flavor
        self.results: Dict[str, Dict[str, Any]] = {}
        self.final_results: Dict[str, Dict[str, Any]] = {}
        self.type_2_table = "その他実績"
        self.expected_headers = ["実績コード", "⼈員", "開始時刻", "終了時刻", "備考"]
    
    def _get_pdf_files(self, input_path: str) -> List[str]:
        """Get list of PDF files from single file or folder."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                return [str(input_path)]
            else:
                raise ValueError(f"File is not a PDF: {input_path}")
        
        elif input_path.is_dir():
            pdf_files = glob.glob(os.path.join(input_path, "*.pdf"))
            if not pdf_files:
                raise ValueError(f"No PDF files found in folder: {input_path}")
            return pdf_files
        
        else:
            raise ValueError(f"Invalid path: {input_path}")
    
    def _parse_filename(self, filepath: str) -> Dict[str, str]:
        """Parse filename structure: date_machine_shift_type.pdf"""
        filename = Path(filepath).stem
        parts = filename.split('_')
        
        if len(parts) != 4:
            raise ValueError(f"Invalid filename format: {filename}. Expected: date_machine_shift_type.pdf")
        
        return {
            'date': parts[0],
            'machine_no': parts[1], 
            'shift': parts[2],
            'pdf_type': parts[3]
        }
    
    def _get_page_count(self, filepath: str) -> int:
        """Get number of pages in PDF file."""
        try:
            doc = fitz.open(filepath)
            page_count = doc.page_count
            doc.close()
            return page_count
        except Exception as e:
            raise ValueError(f"Error reading PDF {filepath}: {e}")
    
    def _extract_tables(self, filepath: str) -> List:
        """Extract tables using camelot."""
        try:
            tables = camelot.read_pdf(filepath, flavor=self.flavor, pages='all')
            return tables
        except Exception as e:
            print(f"Warning: Could not extract tables from {filepath}: {e}")
            return []
    
    # 🔥 NEW METHOD: Extract text from specific page
    def _extract_page_text(self, filepath: str, page_num: int) -> str:
        """
        Extract all text from a specific page using PyMuPDF.
        
        Args:
            filepath (str): Path to PDF file
            page_num (int): Page number (0-indexed)
            
        Returns:
            str: All text content from the page
        """
        try:
            doc = fitz.open(filepath)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")
            return ""
    
    # 🔥 CHANGED METHOD: Now searches page text instead of table content
    def _find_target_pages_from_text(self, filepath: str, target_text: str) -> List[int]:
        """
        Find page numbers where target text appears in page content.
        
        Args:
            filepath (str): Path to PDF file
            target_text (str): Text to search for (e.g., "その他実績")
            
        Returns:
            List[int]: Page numbers containing the target text (1-indexed for camelot)
        """
        target_pages = []
        
        try:
            doc = fitz.open(filepath)
            
            for page_num in range(doc.page_count):
                page_text = self._extract_page_text(filepath, page_num)
                
                if target_text in page_text:
                    target_pages.append(page_num + 1)  # Convert to 1-indexed for camelot
            
            doc.close()
            
        except Exception as e:
            print(f"Error searching for text in {filepath}: {e}")
        
        return target_pages
    
    def _has_expected_headers(self, table_df) -> bool:
        """Check if table has expected headers for continuation pages."""
        header_text = ""
        for row_idx in range(min(3, len(table_df))):
            row_text = " ".join(str(cell) for cell in table_df.iloc[row_idx])
            header_text += row_text + " "
        
        header_text = re.sub(r'\s+', ' ', header_text.replace('\n', ' ')).strip()
        
        found_headers = 0
        for header in self.expected_headers:
            if header in header_text:
                found_headers += 1
        
        return found_headers >= 2
    
    def _get_table_continuation_pages(self, tables: List, start_page: int) -> List[int]:
        """Find all continuation pages for a multi-page table."""
        continuation_pages = [start_page]
        
        all_pages = sorted(list(set(table.page for table in tables)))
        
        try:
            start_idx = all_pages.index(start_page)
        except ValueError:
            return continuation_pages
        
        for i in range(start_idx + 1, len(all_pages)):
            page_num = all_pages[i]
            
            page_tables = [table for table in tables if table.page == page_num]
            
            has_headers = False
            for table in page_tables:
                if self._has_expected_headers(table.df):
                    has_headers = True
                    break
            
            if has_headers:
                continuation_pages.append(page_num)
            else:
                break
        
        return continuation_pages
    
    def _combine_multipage_tables(self, tables: List, page_numbers: List[int]) -> Optional[Any]:
        """Combine tables from multiple pages into one dataset."""
        combined_data = []
        
        for page_num in page_numbers:
            page_tables = [table for table in tables if table.page == page_num]
            
            for table in page_tables:
                df = table.df
                
                if page_num != page_numbers[0]:
                    start_row = 0
                    for row_idx in range(min(3, len(df))):
                        row_text = " ".join(str(cell) for cell in df.iloc[row_idx])
                        if any(header in row_text for header in self.expected_headers):
                            start_row = row_idx + 1
                            break
                    df = df.iloc[start_row:]
                
                for _, row in df.iterrows():
                    if any(str(cell).strip() for cell in row if str(cell).strip() != 'nan'):
                        combined_data.append(row.tolist())
        
        return combined_data
    
    # 🔥 CHANGED METHOD: Now uses text-based page detection
    def _extract_type2_tables(self, filepath: str, tables: List) -> Dict[str, Any]:
        """
        Extract "その他実績" tables from PDF.
        First finds pages with target text, then extracts tables from those pages.
        """
        # 🔥 CHANGED: Now searches page text instead of table content
        target_pages = self._find_target_pages_from_text(filepath, self.type_2_table)
        
        if not target_pages:
            return {"found": False, "pages": [], "data": None}
        
        all_table_groups = []
        processed_pages = set()
        
        for start_page in target_pages:
            if start_page in processed_pages:
                continue
                
            continuation_pages = self._get_table_continuation_pages(tables, start_page)
            processed_pages.update(continuation_pages)
            
            combined_data = self._combine_multipage_tables(tables, continuation_pages)
            
            all_table_groups.append({
                "start_page": start_page,
                "pages": continuation_pages,
                "data": combined_data,
                "total_pages": len(continuation_pages)
            })
        
        return {
            "found": True,
            "table_groups": all_table_groups,
            "total_groups": len(all_table_groups)
        }
    
    def clean_type2_table_data(self, data: List[List]) -> pd.DataFrame:
        """
        Clean and process type2 table data.
        
        Args:
            data (List[List]): Raw table data from camelot
            
        Returns:
            pd.DataFrame: Cleaned dataframe with proper columns
        """
        if not data:
            return pd.DataFrame()
        # 1. we het the ra data as the list of lists
        # there was error become some of the data is getting stored beyond column3
        
        # all elemets are the string 
        # in each row lets merge the elemets to the col 2
               # Step 0: Merge elements beyond column 2 into column 2
        # row_data = []
        # for row in data:
        #     if len(row) > 3:
        #         row[2] = ' '.join(row[2:])
        #         del row[3:]  # Remove elements beyond column 2
        #     row_data.append(row)
        row_data = []
        for row in data:
            if len(row) > 3:
                base = row[2]  # Original column 2 content
                extra = ' '.join(row[3:])  # Content from columns 3 onward
                
                if '〜' in base:
                    # Insert extra content right after '〜'
                    idx = base.find('〜')
                    row[2] = base[:idx+1] + ' ' + extra + base[idx+1:]
                else:
                    # No '〜', just append at the end
                    row[2] = base + ' ' + extra
                
                del row[3:]  # Remove the extra columns
            row_data.append(row)
        print("Aftre merging the data the row data looks like \n ", row_data)   
        
        # merged_data = []
        # ref_row = None  # Initialize ref_row
        # for row in row_data:
        #     if row[0] != '':  # New row starts
        #         if ref_row is not None:
        #             merged_data.append(ref_row)  # Save previous complete row
        #         ref_row = row.copy()  # Start new reference row
        #     elif ref_row is not None:  # Continuation row
        #         ref_row[2] += ' ' + row[2]  # Append to previous row's column 2

        # # Don't forget the last row
        # if ref_row is not None:
        #     merged_data.append(ref_row)

        # Merge multi-line rows
        # Merge multi-line rows
        merged_data = []
        ref_row = None  # Initialize ref_row

        for row in row_data:
            if row[0] != '':  # New row starts
                if ref_row is not None:
                    merged_data.append(ref_row)  # Save previous complete row
                ref_row = row.copy()  # Start new reference row
            elif ref_row is not None:  # Continuation row
                # Find '〜' in ref_row[2] and add row[2] after it
                if '〜' in ref_row[2]:
                    idx = ref_row[2].find('〜')
                    ref_row[2] = ref_row[2][:idx+1] + ' ' + row[2] + ref_row[2][idx+1:]
                else:
                    # If no '〜', append at the end as fallback
                    ref_row[2] += ' ' + row[2]

        # Don't forget the last row
        if ref_row is not None:
            merged_data.append(ref_row)

        data = merged_data

        print("merged data after cleaning the rows : \n", data)
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Step 1: Slice columns - keep only first 3 columns (0, 1, 2)
        if df.shape[1] >= 3:
            sliced_df = df.iloc[:, 0:3].copy()
        else:
            sliced_df = df.copy()
        
        # Rename columns for clarity
        sliced_df.columns = ['実績コード', '人員', '時刻備考'] if sliced_df.shape[1] >= 3 else sliced_df.columns
        
        # Step 2: Filter rows - keep only valid codes
        valid_codes_pattern = r'^\d+:'  # Pattern: number followed by colon
        
        # Filter rows where column 0 matches valid code pattern
        if '実績コード' in sliced_df.columns:
            mask = sliced_df['実績コード'].astype(str).str.match(valid_codes_pattern, na=False)
            filtered_df = sliced_df[mask].copy()
        else:
            filtered_df = sliced_df.copy()
        
        # Step 3: Split '時刻備考' column into 3 columns
        if '時刻備考' in filtered_df.columns and not filtered_df.empty:
            # Initialize new columns
            filtered_df['開始時刻'] = ''
            filtered_df['終了時刻'] = ''
            filtered_df['備考'] = ''
            
            for idx, row in filtered_df.iterrows():
                時刻備考_data = str(row['時刻備考'])
                
                # Skip header rows
                if '開始時刻' in 時刻備考_data or '終了時刻' in 時刻備考_data:
                    continue
                
                # Parse the combined data
                parsed = self._parse_time_and_remarks(時刻備考_data)
                
                filtered_df.loc[idx, '開始時刻'] = parsed['開始時刻']
                filtered_df.loc[idx, '終了時刻'] = parsed['終了時刻']
                filtered_df.loc[idx, '備考'] = parsed['備考']
            
            # Drop the original combined column
            filtered_df = filtered_df.drop('時刻備考', axis=1)
        
        # Reset index
        filtered_df = filtered_df.reset_index(drop=True)
        
        return filtered_df
    

    def _parse_time_and_remarks(self, text: str) -> Dict[str, str]:
        """
        Parse combined time and remarks text.
        
        Args:
            text (str): Combined text like "08:35\n08:45\nポカヨケ\n〜" or "08:25 08:35 ポカヨケ 〜"
            
        Returns:
            Dict[str, str]: Parsed time and remarks
        """
        # Initialize result
        result = {'開始時刻': '', '終了時刻': '', '備考': ''}
        
        if not text or text == 'nan':
            return result
        
        # Remove common PDF encoding artifacts
        text = text.replace('(cid:0)', '').replace('(cid:1)', '').replace('(cid:2)', '')
        text = re.sub(r'\(cid:\d+\)', '', text)
        
        # 🔥 ENHANCED: Remove '〜' and clean
        text = text.replace('〜', '').strip()  # Remove all '〜' and trim
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Try splitting by newlines first
        parts = [part.strip() for part in text.split('\n') if part.strip()]
        
        # If no newlines, fall back to space splitting (for merged text)
        if len(parts) == 1:
            parts = [part.strip() for part in text.split() if part.strip()]
        
        # Time pattern: HH:MM format
        time_pattern = r'^\d{1,2}:\d{2}$'
        
        times_found = []
        remarks_parts = []
        
        for part in parts:
            if re.match(time_pattern, part):
                times_found.append(part)
            else:
                # Skip empty or artifact parts
                if part and part not in ['〜', '']:  # Extra filter for '〜'
                    remarks_parts.append(part)
        
        # Assign times
        if len(times_found) >= 1:
            result['開始時刻'] = times_found[0]
        if len(times_found) >= 2:
            result['終了時刻'] = times_found[1]
        
        # Combine and clean remarks
        if remarks_parts:
            combined_remarks = ' '.join(remarks_parts)
            result['備考'] = re.sub(r'\s+', ' ', combined_remarks).strip()
        
        return result


    
    def process(self, input_path: str) -> Dict[str, Dict[str, Any]]:
        """Main method to process PDF files and extract information."""
        pdf_files = self._get_pdf_files(input_path)
        
        for pdf_path in pdf_files:
            filename = Path(pdf_path).name
            
            try:
                file_info = self._parse_filename(pdf_path)
                file_info['pages'] = self._get_page_count(pdf_path)
                
                tables = self._extract_tables(pdf_path)
                file_info['all_tables'] = tables
                
                # 🔥 CHANGED: Now uses text-based detection
                file_info['type2_tables'] = self._extract_type2_tables(pdf_path, tables)
                
                file_info['filepath'] = pdf_path
                
                self.results[filename] = file_info
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return self.results

    # Updated process method to include cleaning
    def process_and_clean(self, input_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Process PDFs and return cleaned data.
        """
        # First process as before
        results = self.process(input_path)
        
        # Then clean the type2 tables
        for filename, info in results.items():
            type2_tables = info.get('type2_tables', {})
            if type2_tables.get('found'):
                cleaned_groups = []
                
                for group in type2_tables.get('table_groups', []):
                    raw_data = group.get('data', [])
                    # print("raw data for group:-----------------\n", raw_data)
                    # Clean the data
                    cleaned_df = self.clean_type2_table_data(raw_data)
                    # print("\nCleaned data for group:-----------------\n", cleaned_df)
                    # Update group with cleaned data
                    cleaned_group = group.copy()
                    cleaned_group['cleaned_data'] = cleaned_df
                    cleaned_group['raw_data'] = raw_data  # Keep original for reference
                    
                    cleaned_groups.append(cleaned_group)
                
                # Update results with cleaned data
                info['type2_tables']['table_groups'] = cleaned_groups
        
        return results
    
    def _format_for_csv(self, df: pd.DataFrame, info: Dict[str, Any]) -> pd.DataFrame:
        """
        Format dataframe to match required CSV structure with midnight crossover detection.
        """
        if df.empty:
            return df
        
        # Create new dataframe with required columns
        formatted_df = pd.DataFrame()
        
        # Add date column (from filename)
        date_str = info.get('date', '')
        if date_str:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                base_date = date_obj.strftime('%Y/%m/%d')
                # Calculate next day for midnight crossover
                next_day_obj = date_obj + timedelta(days=1)
                next_date = next_day_obj.strftime('%Y/%m/%d')
            except:
                base_date = date_str
                next_date = date_str
        else:
            base_date = ''
            next_date = ''
        
        formatted_df['作業日'] = [base_date] * len(df)
        formatted_df['設備ユニット'] = [info.get('machine_no', '')] * len(df)
        formatted_df['昼夜'] = [info.get('shift', '')] * len(df)
        formatted_df['実績コード'] = df['実績コード'].values
        formatted_df['人員'] = df['人員'].values
        
        # 🔥 NEW: Sequential midnight crossover detection
        start_times = []
        end_times = []
        current_date = base_date  # Track current working date
        
        for idx, row in df.iterrows():
            start_time = row['開始時刻']
            end_time = row['終了時刻']
            
            if start_time and end_time and start_time != '' and end_time != '':
                try:
                    # Parse hours
                    start_hour = int(start_time.split(':')[0])
                    end_hour = int(end_time.split(':')[0])
                    
                    # Check if we've crossed midnight by comparing with previous row
                    if idx > 0:
                        prev_time = df.iloc[idx-1]['終了時刻']
                        if prev_time and prev_time != '':
                            prev_hour = int(prev_time.split(':')[0])
                            
                            # If previous time was late (>= 20) and current start is early (<= 6)
                            # Then we've crossed midnight
                            if prev_hour >= 20 and start_hour <= 6:
                                current_date = next_date
                    
                    # Assign dates
                    start_times.append(f"{current_date} {start_time}")
                    
                    # For end time, check if it crosses midnight within this record
                    if start_hour >= 22 and end_hour <= 6:
                        end_times.append(f"{next_date} {end_time}")
                    else:
                        end_times.append(f"{current_date} {end_time}")
                        
                except (ValueError, IndexError):
                    # If parsing fails, use current date for both
                    start_times.append(f"{current_date} {start_time}" if start_time else '')
                    end_times.append(f"{current_date} {end_time}" if end_time else '')
            else:
                # Handle empty times
                start_times.append(f"{current_date} {start_time}" if start_time else '')
                end_times.append(f"{current_date} {end_time}" if end_time else '')
        
        formatted_df['開始時刻'] = start_times
        formatted_df['終了時刻'] = end_times
        formatted_df['備考'] = df['備考'].values
        
        return formatted_df

    

    def save_to_csv(self, results: Dict[str, Dict[str, Any]], output_folder: str = "output") -> None:
        """
        Save cleaned table data to CSV files with required format.
        
        Args:
            results: Processed results from process_and_clean
            output_folder: Folder to save CSV files
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        for filename, info in results.items():
            type2_tables = info.get('type2_tables', {})
            
            if not type2_tables.get('found'):
                print(f"No type2 tables found in {filename}")
                continue
            
            # Combine all table groups into one dataframe
            all_cleaned_data = []
            
            for group in type2_tables.get('table_groups', []):
                cleaned_df = group.get('cleaned_data')
                if cleaned_df is not None and not cleaned_df.empty:
                    all_cleaned_data.append(cleaned_df)
            
            if not all_cleaned_data:
                print(f"No cleaned data found in {filename}")
                continue
            
            # Combine all groups into one dataframe
            combined_df = pd.concat(all_cleaned_data, ignore_index=True)
            
            # Add required columns
            final_df = self._format_for_csv(combined_df, info)
            
            # Create CSV filename (same as PDF but with .csv extension)
            csv_filename = filename.replace('.pdf', '.csv')
            csv_filename = "Table_2_" + csv_filename  # Prefix with Table_2_
            csv_path = os.path.join(output_folder, csv_filename)
            
            # Save to CSV
            final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Saved: {csv_path}")
    
    def _create_final_result(self, filename: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create clean final result with only essential data.
        
        Args:
            filename: PDF filename
            info: Full processing info
            
        Returns:
            Dict: Clean result with only necessary data
        """
        final_info = {
            # File metadata (from filename parsing)
            'date': info.get('date', ''),
            'machine_no': info.get('machine_no', ''), 
            'shift': info.get('shift', ''),
            'pdf_type': info.get('pdf_type', ''),
            'filepath': info.get('filepath', ''),
            'total_pages': info.get('pages', 0),
            
            # Only cleaned table data
            'type2_found': False,
            'cleaned_dataframes': [],
            'total_records': 0
        }
        
        # Process type2 tables if found
        type2_tables = info.get('type2_tables', {})
        if type2_tables.get('found'):
            final_info['type2_found'] = True
            
            for group in type2_tables.get('table_groups', []):
                cleaned_df = group.get('cleaned_data')
                if cleaned_df is not None and not cleaned_df.empty:
                    # Format the dataframe for final use
                    formatted_df = self._format_for_csv(cleaned_df, info)
                    
                    final_info['cleaned_dataframes'].append({
                        'pages': group.get('pages', []),
                        'dataframe': formatted_df,
                        'record_count': len(formatted_df)
                    })
                    
                    final_info['total_records'] += len(formatted_df)
        
        return final_info


    # Updated main processing method
    def process_clean_and_save(self, input_path: str, output_folder: str = "output") -> Dict[str, Dict[str, Any]]:
        """
        Complete pipeline: process PDFs, clean data, and save to CSV.
        
        Args:
            input_path: Path to PDF file or folder
            output_folder: Output folder for CSV files
            
        Returns:
            Dict: Processing results
        """
        # Step 1: Process and clean
        results = self.process_and_clean(input_path)
        
        # Step 2: Save to CSV
        self.save_to_csv(results, output_folder)
        
        # Step 2: Create clean final results
        self.final_results = {}
        for filename, info in results.items():
            self.final_results[filename] = self._create_final_result(filename, info)

        results = self.final_results
        self.results = self.final_results
        return results

    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get processed results."""
        return self.results
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = {}

    def print_summary(self) -> None:
        """Print summary of final results."""
        if not self.final_results:
            print("No final results available. Run process_clean_and_save() first.")
            return
        
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        
        for filename, info in self.final_results.items():
            print(f"📄 {filename}")
            print(f"   📅 Date: {info['date']}")
            print(f"   🏭 Machine: {info['machine_no']}")
            if info['shift'] == '昼':
                print(f"   🌞 Shift: {info['shift']}")
            else:
                print(f"   🌙 Shift: {info['shift']}")
            print(f"   📋 Team: {info['pdf_type']}")
            print(f"   📄 Pages: {info['total_pages']}")
            print(f"   ✅ Type2 Found: {info['type2_found']}")
            print(f"   📊 Total Records: {info['total_records']}")
            
            if info['cleaned_dataframes']:
                for i, df_info in enumerate(info['cleaned_dataframes']):
                    print(f"      📋 Table {i+1}: {df_info['record_count']} records from pages {df_info['pages']}")
            
            print("-" * 50)

if __name__ == "__main__":
    extractor = PDFTableExtractor(flavor='lattice')
    input_path = "2025-11-17_10-1615_昼_B.pdf"
    input_path2 = "test_folder"
    # results = extractor.process_and_clean(input_path)

    results = extractor.process_clean_and_save(input_path2, "output")













