
// Update the Tessa chat message handler to support report downloads
// Add this code after: addMessage(data.response || 'Sorry...', false, false, data);

if (data.data && (data.data.download_urls || data.data.download_url)) {
    const lastMessage = document.querySelector('.tessa-message:last-child .tessa-message-content');
    if (lastMessage) {
        const downloadContainer = document.createElement('div');
        downloadContainer.className = 'mt-3';
        downloadContainer.style.cssText = 'margin-top: 15px;';
        
        // Single download
        if (data.data.download_url) {
            const downloadBtn = document.createElement('a');
            downloadBtn.href = data.data.download_url;
            downloadBtn.className = 'btn btn-primary';
            downloadBtn.style.cssText = 'background: #667eea; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block;';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Report';
            downloadContainer.appendChild(downloadBtn);
        }
        
        // Multiple downloads (PDF and Excel)
        if (data.data.download_urls) {
            if (data.data.download_urls.pdf) {
                const pdfBtn = document.createElement('a');
                pdfBtn.href = data.data.download_urls.pdf;
                pdfBtn.className = 'btn btn-danger';
                pdfBtn.style.cssText = 'background: #dc3545; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block; margin-right: 10px;';
                pdfBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Download PDF';
                downloadContainer.appendChild(pdfBtn);
            }
            
            if (data.data.download_urls.excel) {
                const excelBtn = document.createElement('a');
                excelBtn.href = data.data.download_urls.excel;
                excelBtn.className = 'btn btn-success';
                excelBtn.style.cssText = 'background: #28a745; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block;';
                excelBtn.innerHTML = '<i class="fas fa-file-excel"></i> Download Excel';
                downloadContainer.appendChild(excelBtn);
            }
        }
        
        lastMessage.appendChild(downloadContainer);
    }
}
