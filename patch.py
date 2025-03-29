    # This completes the truncated batch_test method
    def batch_test(self):
        """Test multiple PDF forms in batch mode."""
        if not self.model or not self.processor:
            messagebox.showwarning("Warning", "LayoutLMv3 model not loaded")
            return
        
        # Ask for directory containing PDFs
        dir_path = filedialog.askdirectory(title="Select Directory with PDF Forms")
        if not dir_path:
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        # Find all PDF files
        pdf_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            messagebox.showinfo("Info", "No PDF files found in the selected directory.")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Processing PDF files in batch mode...").pack(padx=10, pady=10)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(pdf_files))
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        status_label = ttk.Label(progress_window, text="Starting...")
        status_label.pack(padx=10, pady=5)
        
        # Function to process PDFs in batch
        def process_batch():
            results = {}
            
            for i, pdf_file in enumerate(pdf_files):
                # Update progress
                filename = os.path.basename(pdf_file)
                progress_var.set(i)
                status_label.config(text=f"Processing {filename}...")
                progress_window.update_idletasks()
                
                try:
                    # Process PDF
                    pdf_results = self.process_pdf_for_batch(pdf_file)
                    results[pdf_file] = pdf_results
                    
                    # Write individual results file
                    output_file = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(filename)[0]}_results.json"
                    )
                    
                    with open(output_file, 'w') as f:
                        json.dump(pdf_results, f, indent=4)
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")
                    results[pdf_file] = {"error": str(e)}
            
            # Write summary file
            summary_file = os.path.join(output_dir, "batch_summary.json")
            with open(summary_file, 'w') as f:
                summary = {
                    "total_files": len(pdf_files),
                    "processed_at": datetime.now().isoformat(),
                    "results": {os.path.basename(k): v for k, v in results.items()}
                }
                json.dump(summary, f, indent=4)
            
            # Close progress window
            progress_window.destroy()
            
            # Show completion message
            messagebox.showinfo(
                "Batch Complete", 
                f"Processed {len(pdf_files)} PDF files. Results saved to {output_dir}"
            )
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()
    
    # Add the missing process_pdf_for_batch method
    def process_pdf_for_batch(self, pdf_file):
        """Process a PDF file for batch testing."""
        results = {"pages": {}}
        
        # Open PDF
        doc = fitz.open(pdf_file)
        
        # Process each page
        for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages
            page = doc[page_num]
            
            # Extract text and boxes
            words, boxes = self.extract_text_from_page(page)
            
            if not words:
                results["pages"][str(page_num)] = {"error": "No text found"}
                continue
            
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Process with LayoutLMv3
            encoding = self.processor(
                img,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Extract field values
            field_values = {}
            current_field = None
            current_text = ""
            
            for i, (word, pred) in enumerate(zip(words, predictions)):
                label = self.id_to_label.get(pred, "O")
                
                # Check if this is a beginning of a field
                if label.startswith("B-"):
                    # Save the previous field if any
                    if current_field and current_text:
                        if current_field not in field_values:
                            field_values[current_field] = []
                        field_values[current_field].append(current_text.strip())
                    
                    # Start new field
                    current_field = label[2:]  # Remove "B-" prefix
                    current_text = word
                # Check if this is inside a field
                elif label.startswith("I-") and current_field == label[2:]:
                    current_text += " " + word
                # Check if this is outside any field
                elif label == "O":
                    # Save the previous field if any
                    if current_field and current_text:
                        if current_field not in field_values:
                            field_values[current_field] = []
                        field_values[current_field].append(current_text.strip())
                    
                    # Reset
                    current_field = None
                    current_text = ""
            
            # Save the last field if any
            if current_field and current_text:
                if current_field not in field_values:
                    field_values[current_field] = []
                field_values[current_field].append(current_text.strip())
            
            # Store results for this page
            results["pages"][str(page_num)] = field_values
        
        doc.close()
        return results
