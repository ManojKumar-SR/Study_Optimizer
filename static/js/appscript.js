let postureTimer = null;
let emotionTimer = null;
let distractionTimer = null;
let engagementTimer = null;

function startCamera() {
    document.getElementById("videoFeed").src = "/start_camera";
    document.getElementById("videoFeed").style.display = "block";
    document.getElementById("alertBox").classList.remove("hidden");
    document.getElementById("engagementStatus").innerText = "Engagement Status: Loading...";
    document.getElementById("emotionStatus").innerText = "Emotion: Loading...";
    document.getElementById("distractionStatus").innerText = "Distraction Status: Loading...";
    document.getElementById("postureStatus").innerText = "Posture Status: Loading...";
    fetchStatus();
}

function stopCamera() {
    fetch("/stop_camera", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
            document.getElementById("videoFeed").src = "";
            document.getElementById("videoFeed").style.display = "none";
            document.getElementById("alertBox").classList.add("hidden");
            clearTimers(); // Clear all timers
            document.getElementById("engagementStatus").innerText = "Engagement Status: Not Engaged";
            document.getElementById("emotionStatus").innerText = "Emotion: None";
            document.getElementById("distractionStatus").innerText = "Distraction Status: Not Distracted";
            document.getElementById("postureStatus").innerText = "Posture Status: Unknown";
        });
}

function fetchStatus() {
    setInterval(() => {
        fetch("/status")
            .then(response => response.json())
            .then(data => {
                updateStatus("engagement", data.engagement_status);
                updateStatus("emotion", data.dominant_emotion);
                updateStatus("distraction", data.distraction_status);
                updateStatus("posture", data.posture_status);
            })
            .catch(error => console.error("Error fetching status:", error));
    }, 1000);
}

function updateStatus(type, status) {
    if (type === "engagement") {
        document.getElementById("engagementStatus").innerText = "Engagement Status: " + status;
        if (status === "Not Engaged") {
            startTimer("engagement", 10, "Please stay focused.");
        } else {
            clearTimer("engagement");
        }
    }

    if (type === "emotion") {
        document.getElementById("emotionStatus").innerText = "Emotion: " + status;
        if (status.includes("Sad") || status.includes("Angry")) {
            startTimer("emotion", 30, "You seem stressed. Take a short break.");
        } else {
            clearTimer("emotion");
        }
    }

    if (type === "distraction") {
        document.getElementById("distractionStatus").innerText = "Distraction Status: " + status;
        if (status === "Distracted") {
            startTimer("distraction", 10, "Please focus. Avoid distractions like mobile phones.");
        } else {
            clearTimer("distraction");
        }
    }

    if (type === "posture") {
        document.getElementById("postureStatus").innerText = "Posture Status: " + status;
        if (status.includes("Bad Posture")) {
            startTimer("posture", 10, "Maintain a good posture for comfort.");
        } else {
            clearTimer("posture");
        }
    }
}

// Helper function to start a timer for each condition
function startTimer(type, threshold, alertMessage) {
    if (!window[`${type}Timer`]) {
        window[`${type}Timer`] = setTimeout(() => {
            addAlert(alertMessage);
        }, threshold * 1000);
    }
}

// Helper function to clear a timer for each condition
function clearTimer(type) {
    if (window[`${type}Timer`]) {
        clearTimeout(window[`${type}Timer`]);
        window[`${type}Timer`] = null;
    }
}

// Clear all timers when stopping the camera
function clearTimers() {
    clearTimer("engagement");
    clearTimer("emotion");
    clearTimer("distraction");
    clearTimer("posture");
}

// Add alert message to the alert box
function addAlert(message) {
    const alertBoxContent = document.getElementById("alertBoxContent");
    const newAlert = document.createElement("p");
    newAlert.classList.add("mb-2", "text-red-600");
    newAlert.textContent = message;
    alertBoxContent.appendChild(newAlert);
    
    // Optionally scroll to the latest alert
    alertBoxContent.scrollTop = alertBoxContent.scrollHeight;
}
