document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    const chatMessages = document.getElementById('chat-messages');
    const contextSection = document.getElementById('context-section');
    const contextContainer = document.getElementById('context-container');
    
    // Check if the chat interface is available
    if (!questionInput || !askButton || !chatMessages) {
        return; // Exit if we're not on the right page
    }
    
    // Event listener for submit button
    askButton.addEventListener('click', sendQuestion);
    
    // Event listener for Enter key in input
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendQuestion();
        }
    });
    
    // Function to send question to backend
    function sendQuestion() {
        const question = questionInput.value.trim();
        
        if (!question) {
            return; // Don't send empty questions
        }
        
        // Disable input and button while processing
        questionInput.disabled = true;
        askButton.disabled = true;
        
        // Add user message to chat
        addMessage(question, 'user');
        
        // Add loading indicator
        const loadingMessageId = 'loading-' + Date.now();
        addSystemMessage('<div class="spinner me-2"></div> Thinking...', loadingMessageId);
        
        // Scroll to bottom
        scrollToBottom();
        
        // Clear input
        questionInput.value = '';
        
        // Send question to backend
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            removeMessage(loadingMessageId);
            
            if (data.success) {
                // Add bot's answer to chat
                addMessage(data.answer, 'bot');
                
                // Show retrieval techniques used if available
                if (data.techniques_used && data.techniques_used.length > 0) {
                    const techniquesHtml = '<div class="techniques-info"><i class="fas fa-info-circle me-2"></i>Techniques used: ' + 
                        data.techniques_used.join(', ') + '</div>';
                    addSystemMessage(techniquesHtml);
                }
                
                // Show contexts if available
                if (data.contexts && data.contexts.length > 0) {
                    displayContexts(data.contexts);
                }
            } else {
                // Handle error
                addSystemMessage('<i class="fas fa-exclamation-triangle me-2"></i> ' + data.error);
            }
        })
        .catch(error => {
            // Remove loading message
            removeMessage(loadingMessageId);
            
            // Show error message
            addSystemMessage('<i class="fas fa-exclamation-triangle me-2"></i> Error: Could not connect to server.');
            console.error('Error:', error);
        })
        .finally(() => {
            // Re-enable input and button
            questionInput.disabled = false;
            askButton.disabled = false;
            questionInput.focus();
            
            // Scroll to bottom
            scrollToBottom();
        });
    }
    
    // Function to add a message to the chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        
        if (sender === 'user') {
            messageDiv.classList.add('user-message');
            messageDiv.innerHTML = `<div>${escapeHtml(text)}</div>`;
        } else if (sender === 'bot') {
            messageDiv.classList.add('bot-message');
            messageDiv.innerHTML = `<div class="text-pre-wrap">${formatBotResponse(text)}</div>`;
        }
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Function to add a system message
    function addSystemMessage(html, id = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('system-message');
        if (id) {
            messageDiv.id = id;
        }
        messageDiv.innerHTML = html;
        chatMessages.appendChild(messageDiv);
    }
    
    // Function to remove a message by ID
    function removeMessage(id) {
        const messageToRemove = document.getElementById(id);
        if (messageToRemove) {
            messageToRemove.remove();
        }
    }
    
    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to display contexts
    function displayContexts(contexts) {
        // Clear previous contexts
        contextContainer.innerHTML = '';
        
        // Add each context as a card
        contexts.forEach((context, index) => {
            const contextDiv = document.createElement('div');
            contextDiv.classList.add('card', 'context-card', 'mb-3');
            
            let badgeColor = 'success';
            if (context.score < 70) badgeColor = 'warning';
            if (context.score < 50) badgeColor = 'danger';
            
            // Get method icon and description
            let methodIcon = 'fa-search';
            let methodName = 'standard';
            
            if (context.method) {
                if (context.method === 'multi_query') {
                    methodIcon = 'fa-project-diagram';
                    methodName = 'Multi-Query';
                } else if (context.method === 'step_back') {
                    methodIcon = 'fa-arrow-up';
                    methodName = 'Step-Back';
                } else if (context.method === 'adaptive') {
                    methodIcon = 'fa-sliders-h';
                    methodName = 'Adaptive';
                }
            }
            
            contextDiv.innerHTML = `
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span>
                        <i class="fas fa-file-alt me-2"></i> Context ${index + 1}
                        <span class="badge bg-secondary ms-2"><i class="fas ${methodIcon} me-1"></i> ${methodName}</span>
                    </span>
                    <span class="badge bg-${badgeColor} relevance-badge">Relevance: ${context.score}%</span>
                </div>
                <div class="card-body">
                    <p class="card-text text-pre-wrap">${escapeHtml(context.content)}</p>
                </div>
            `;
            
            contextContainer.appendChild(contextDiv);
        });
        
        // Show the context section
        contextSection.classList.remove('d-none');
    }
    
    // Helper function to escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Function to format bot response (handling code blocks, lists, etc.)
    function formatBotResponse(text) {
        // Basic markdown-like formatting
        let formattedText = escapeHtml(text);
        
        // Format code blocks
        formattedText = formattedText.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
        formattedText = formattedText.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Format bold text
        formattedText = formattedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Format bullet lists
        formattedText = formattedText.replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>');
        formattedText = formattedText.replace(/<li>(.+)<\/li>/g, '<ul><li>$1</li></ul>');
        formattedText = formattedText.replace(/<\/ul>\s*<ul>/g, '');
        
        return formattedText;
    }
    
    // Delete document confirmation
    document.querySelectorAll('.delete-form').forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!confirm('Are you sure you want to delete this document?')) {
                e.preventDefault();
            }
        });
    });
    
    // Document processing progress
    const processForm = document.getElementById('process-documents-form');
    const processBtn = document.getElementById('process-btn');
    const progressContainer = document.getElementById('processing-progress-container');
    const progressBar = document.getElementById('processing-progress-bar');
    const statusText = document.getElementById('processing-status');
    
    if (processForm && progressBar) {
        processForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show progress bar at 0%
            progressContainer.classList.remove('d-none');
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            progressBar.textContent = '0%';
            
            // Show initial status
            statusText.classList.remove('d-none');
            statusText.textContent = 'Starting document processing...';
            
            // Disable button
            processBtn.disabled = true;
            
            // Submit the form via AJAX
            fetch(processForm.action, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Start polling for progress updates
                    pollProcessingProgress();
                } else {
                    // Show error
                    statusText.textContent = data.error || 'Error processing documents';
                    statusText.classList.add('text-danger');
                    processBtn.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                statusText.textContent = 'Error starting document processing';
                statusText.classList.add('text-danger');
                processBtn.disabled = false;
            });
        });
    }
    
    // Function to poll for processing progress
    function pollProcessingProgress() {
        const pollInterval = setInterval(() => {
            fetch('/processing-status', {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update progress bar
                    const progress = data.progress || 0;
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                    progressBar.textContent = `${progress}%`;
                    
                    // Update status text
                    statusText.textContent = data.message || 'Processing...';
                    
                    // If complete, stop polling and enable button
                    if (data.complete) {
                        clearInterval(pollInterval);
                        processBtn.disabled = false;
                        
                        // Reload page after short delay to update the UI
                        setTimeout(() => {
                            window.location.reload();
                        }, 1000);
                    }
                } else {
                    // Error occurred, stop polling
                    clearInterval(pollInterval);
                    statusText.textContent = data.error || 'Error during processing';
                    statusText.classList.add('text-danger');
                    processBtn.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                clearInterval(pollInterval);
                statusText.textContent = 'Error checking processing status';
                statusText.classList.add('text-danger');
                processBtn.disabled = false;
            });
        }, 1000); // Poll every second
    }
});
// Add upload-during-chat functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create upload button
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    
    if (!questionInput || !askButton) {
        return; // Not on chat page
    }
    
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
    
    // Add the elements to the page
    const askButtonParent = askButton.parentNode;
    askButtonParent.insertBefore(uploadButton, askButton);
    document.body.appendChild(fileInput);
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            
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
                    
                    // Enable the input and ask button if they were disabled
                    questionInput.disabled = false;
                    askButton.disabled = false;
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
            
            // Clear the file input to allow uploading the same file again
            fileInput.value = '';
        }
    });
    
    // Handle upload button click
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });
});
