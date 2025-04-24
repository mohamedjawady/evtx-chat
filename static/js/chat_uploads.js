
/**
 * JavaScript to handle document uploads during chat conversations
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a page with chat functionality
    const chatMessages = document.getElementById('chat-messages');
    const questionInput = document.getElementById('question-input');
    const askButtonContainer = questionInput ? questionInput.nextElementSibling : null;
    
    if (!chatMessages || !questionInput || !askButtonContainer) {
        return; // Not on chat page
    }
    
    // Create upload button
    const uploadButton = document.createElement('button');
    uploadButton.className = 'btn btn-outline-secondary';
    uploadButton.type = 'button';
    uploadButton.title = 'Upload Document';
    uploadButton.innerHTML = '<i class="fas fa-paperclip"></i>';
    
    // Create file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'chat-file-upload';
    fileInput.accept = '.pdf,.evtx';
    fileInput.style.display = 'none';
    
    // Add them to the DOM
    askButtonContainer.parentNode.insertBefore(uploadButton, askButtonContainer);
    document.body.appendChild(fileInput);
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            uploadDocumentInChat(file);
        }
    });
    
    // Handle upload button click
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Function to upload document in chat
    function uploadDocumentInChat(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const loadingMessageId = 'upload-' + Date.now();
        addSystemMessage(`<div class="spinner me-2"></div> Uploading and processing ${file.name}...`, loadingMessageId);
        
        fetch('/upload-in-chat', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            removeMessage(loadingMessageId);
            
            if (data.success) {
                addSystemMessage(`<i class="fas fa-check-circle me-2"></i> ${data.message}`);
            } else {
                addSystemMessage(`<i class="fas fa-exclamation-triangle me-2"></i> ${data.error}`);
            }
        })
        .catch(error => {
            removeMessage(loadingMessageId);
            addSystemMessage('<i class="fas fa-exclamation-triangle me-2"></i> Error uploading document. Please try again.');
            console.error('Error:', error);
        });
    }
});
