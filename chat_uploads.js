/**
 * JavaScript to handle document uploads during chat conversations
 * 
 * Instructions: Include this file in the HTML template after main.js
 */

document.addEventListener('DOMContentLoaded', function() {
    // Wait for main.js to initialize
    setTimeout(function() {
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
        
        // Adjust input group for button
        const inputGroup = questionInput.parentNode;
        inputGroup.style.display = 'flex';
        
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
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            // Show upload message
            const loadingMessageId = 'upload-' + Date.now();
            addSystemMessage(`<div class="spinner me-2"></div> Uploading and processing ${file.name}...`, loadingMessageId);
            
            // Upload file
            fetch('/upload-in-chat', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                removeMessage(loadingMessageId);
                
                if (data.success) {
                    // Add success message
                    addSystemMessage(`<i class="fas fa-check-circle me-2"></i> ${data.message}`);
                } else {
                    // Add error message
                    addSystemMessage(`<i class="fas fa-exclamation-triangle me-2"></i> ${data.error}`);
                }
            })
            .catch(error => {
                // Remove loading message
                removeMessage(loadingMessageId);
                
                // Add error message
                addSystemMessage('<i class="fas fa-exclamation-triangle me-2"></i> Error uploading document. Please try again.');
                console.error('Error:', error);
            });
        }
        
        // Helper function to add system message (should be defined in main.js, but adding fallback)
        function addSystemMessage(html, id = null) {
            if (typeof window.addSystemMessage === 'function') {
                window.addSystemMessage(html, id);
                return;
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('system-message');
            if (id) {
                messageDiv.id = id;
            }
            messageDiv.innerHTML = html;
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom if scrollToBottom function exists
            if (typeof window.scrollToBottom === 'function') {
                window.scrollToBottom();
            } else {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        
        // Helper function to remove a message (should be defined in main.js, but adding fallback)
        function removeMessage(id) {
            if (typeof window.removeMessage === 'function') {
                window.removeMessage(id);
                return;
            }
            
            const messageToRemove = document.getElementById(id);
            if (messageToRemove) {
                messageToRemove.remove();
            }
        }
        
        // Expose some functions to global scope for use by the main script
        window.uploadDocumentInChat = uploadDocumentInChat;
    }, 100); // Small delay to ensure main.js has initialized
});
