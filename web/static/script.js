document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const showThinkingCheckbox = document.getElementById('show-thinking');
    
    let threadId = null;
    let ws = null;
    
    function initWebSocket() {
        if (ws) {
            ws.close();
        }
        
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.thread_id) {
                threadId = data.thread_id;
            }
            
            if (data.history) {
                data.history.forEach(message => {
                    addMessage(message.role, message.content);
                });
            }
            
            if (data.final_response) {
                addMessage('assistant', data.final_response);
            }
            
            if (data.error) {
                addMessage('assistant', `错误: ${data.error}`);
            }
        };
        
        ws.onclose = function() {
            console.log('WebSocket connection closed');
            setTimeout(initWebSocket, 1000);
        };
    }
    
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        if (role === 'tool') {
            messageDiv.className = 'message thinking-message';
            messageDiv.innerHTML = `<strong>思考:</strong> ${content}`;
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        addMessage('user', message);
        messageInput.value = '';
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                message: message,
                thread_id: threadId,
                show_thinking: showThinkingCheckbox.checked
            }));
        }
    }
    
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    showThinkingCheckbox.addEventListener('change', function() {
        // 切换思考过程显示状态
    });
    
    initWebSocket();
});