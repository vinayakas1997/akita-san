import streamlit as st
import os
from pathlib import Path
import io
import contextlib

# Import your PDF extractor class
from pdf_csv import PDFTableExtractor

st.set_page_config(
    page_title="PDF Table Extractor", 
    page_icon="📊",
    layout="wide"
)

# Main heading
st.title("📊 Type2 table PDF ---> CSV")
st.divider()

# Initialize session state for extractor
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
    st.session_state.processed = False

# Create three columns for layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("📁 Input Selection")
    
    # Selection method
    selection_type = st.radio(
        "Select input type:",
        ["📄 Single File", "📁 Folder"],
        horizontal=True
    )
    
    if selection_type == "📄 Single File":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Select a single PDF file to process"
        )
        input_path = uploaded_file
        
    else:  # Folder selection with browse
        # Note: Streamlit doesn't have direct folder picker, using text input
        st.write("📁 **Select Folder:**")
        folder_path = st.text_input(
            "Folder Path",
            placeholder="Enter folder path containing PDF files",
            help="Copy and paste the full path to folder containing PDF files",
            label_visibility="collapsed"
        )
        
        # Add helpful instructions
        st.info("💡 Copy folder path from file explorer (e.g., C:/Users/Documents/PDFs)")
        
        input_path = folder_path if folder_path else None

with col2:
    st.subheader("⚙️ Settings")
    
    # # Input name
    # input_name = st.text_input(
    #     "📝 Input Name",
    #     placeholder="Enter a name for this processing job",
    #     help="Optional: Give a name to identify this processing batch"
    # )
    
    # Output folder path selection
    st.write("📂 **Output Settings:**")
    
    output_option = st.radio(
        "Choose output method:",
        ["📁 Default Output Folder", "🎯 Custom Path"],
        horizontal=True
    )
    
    if output_option == "📁 Default Output Folder":
        output_folder = st.text_input(
            "Folder Name",
            value="output",
            help="Folder will be created in current directory",
            label_visibility="collapsed"
        )
    else:
        output_folder = st.text_input(
            "Full Output Path",
            placeholder="Enter full path for output folder",
            help="e.g., C:/Users/Documents/Output",
            label_visibility="collapsed"
        )

with col3:
    st.subheader("🚀 Action")
    
    # Process button
    process_btn = st.button(
        "🔄 Process PDFs",
        type="primary",
        use_container_width=True
    )

# Processing section
if process_btn:
    if input_path is None:
        st.error("❌ Please select a file or enter folder path!")
    else:
        try:
            with st.spinner("🔄 Processing PDFs..."):
                # Initialize extractor (using lattice as default)
                st.session_state.extractor = PDFTableExtractor(flavor='lattice')
                
                if selection_type == "📄 Single File":
                    # Save uploaded file with original name (not temp prefix)
                    temp_path = uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file
                    results = st.session_state.extractor.process_clean_and_save(temp_path, output_folder)
                    st.success(f"✅ Processed file: {uploaded_file.name}")
                    st.session_state.processed = True
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                else:
                    # Process folder
                    results = st.session_state.extractor.process_clean_and_save(folder_path, output_folder)
                    st.success(f"✅ Processed folder: {folder_path}")
                    st.session_state.processed = True
                
        except Exception as e:
            st.error(f"❌ Error processing: {str(e)}")

# Summary section
if st.session_state.processed:
    st.divider()
    st.subheader("📊 処理の概要")
    
    # Create summary container
    summary_container = st.container()
    
    with summary_container:
        # Capture print_summary output
        if st.session_state.extractor:
            # Capture print output
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                st.session_state.extractor.print_summary()
            summary_text = f.getvalue()
            
            # Display captured summary
            st.code(summary_text)
    
    # Download section
    st.subheader("💾 結果: ファイルを表示 ")
    
    if os.path.exists(output_folder):
        csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
        
        if csv_files:
            st.success(f"✅ Generated {len(csv_files)} CSV files in '{output_folder}' folder")
            
            # Show files list
            with st.expander("📋 Generated Files"):
                for file in csv_files:
                    st.write(f"📄 {file}")
        else:
            st.warning("⚠️ No CSV files generated")
    
    # Reset button
    if st.button("🔄 Process New Files", use_container_width=True):
        st.session_state.processed = False
        st.session_state.extractor = None
        st.rerun()

# Footer
st.divider()
st.caption("🏭 SOMIC Industrial Automation - PDF Table Extractor")

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    
    st.markdown("""
    **How to use:**
    
    1. **Select Input**:
       - 単一のファイルまたはフォルダを選択
       - PDFをアップロードするか、フォルダパスをコピーします
    
    2. **Configure**:
       - セット output folder 名前
       - output フォルダパスをコピーします
    
    3. **プロセス**:
       - 「PDFを処理」をクリック
       - 完了するまで待ちます
    
    4. **確認**:
       - 処理の概要を確認する
       - CSVファイルをダウンロードする
    
    **想定されるファイル形式:**
    `date_machine_shift_type.pdf`
    
    Example: `2025-11-17_10-1615_夜_A.pdf`
    """)
    
    # st.markdown("---")
    # st.markdown("**Settings:**")
    # st.markdown("- **Lattice**: Tables with visible borders")
    # st.markdown("- **Stream**: Tables without borders")