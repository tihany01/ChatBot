const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");

let userMessage;
const API_URL = "http://localhost:5000/api/generar_respuesta";

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
}

const handleChat = () => {
    userMessage = chatInput.value.trim();
    if (!userMessage) return;
    chatInput.value = "";

    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    setTimeout(() => {
        const incomingChatLi = createChatLi("Pensando...", "incoming");
        chatbox.appendChild(incomingChatLi);

        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({consulta: userMessage})
        }).then(response => response.json())
          .then(data => incomingChatLi.querySelector("p").textContent = data.respuesta)
          .catch((error) => {
            incomingChatLi.querySelector("p").textContent = "¡Ups! Algo salió mal. Inténtalo de nuevo.";
          }).finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
    }, 600);
}

sendChatBtn.addEventListener("click", handleChat);
