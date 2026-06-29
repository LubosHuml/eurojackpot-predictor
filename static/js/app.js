// Global state
let currentProfile = 'conservative';
let predictionsData = null;
let pollIntervalId = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchMetrics();
    fetchPredictions();
});

// Fetch system status
async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        
        // Update overview cards
        document.getElementById("stat-latest-date").textContent = data.latest_draw_date;
        document.getElementById("stat-total-draws").textContent = data.total_draws;
        
        // Update status badges
        const dbBadge = document.getElementById("badge-db");
        dbBadge.className = "badge ready";
        dbBadge.querySelector(".badge-label").textContent = `DB Synced (${data.total_draws} draws)`;
        
        const modelBadge = document.getElementById("badge-model");
        if (data.training_state === "Training") {
            modelBadge.className = "badge syncing";
            modelBadge.querySelector(".badge-label").textContent = "Model: Training...";
            showTrainingProgress(true);
        } else if (data.training_state === "Ready") {
            modelBadge.className = "badge ready";
            modelBadge.querySelector(".badge-label").textContent = "Model: Active & Pruned";
            showTrainingProgress(false);
        } else {
            modelBadge.className = "badge untrained";
            modelBadge.querySelector(".badge-label").textContent = "Model: Untrained";
            showTrainingProgress(false);
        }

        // Disable train button if already training
        const btnTrain = document.getElementById("btn-train");
        btnTrain.disabled = (data.training_state === "Training");
        
        // If training has completed and we were polling, refresh predictions
        if (pollIntervalId && data.training_state !== "Training") {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
            showToast("Model retraining completed successfully!");
            fetchMetrics();
            fetchPredictions();
        }
        
        // Start polling if model is training in background
        if (data.training_state === "Training" && !pollIntervalId) {
            pollIntervalId = setInterval(fetchStatus, 3000);
        }
        
    } catch (err) {
        console.error("Error fetching status:", err);
    }
}

// Fetch backtesting metrics
async function fetchMetrics() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        
        if (data.error) {
            document.querySelectorAll(".metric-val").forEach(el => el.textContent = "N/A");
            return;
        }
        
        // Update table values
        document.getElementById("metric-sum-mae").textContent = `${data.sum_mae.toFixed(2)} numbers`;
        document.getElementById("metric-counts-mae").textContent = data.counts_mae.toFixed(4);
        document.getElementById("metric-main-top5").textContent = `${(data.main_top5 * 100).toFixed(1)}%`;
        document.getElementById("metric-main-top10").textContent = `${(data.main_top10 * 100).toFixed(1)}%`;
        document.getElementById("metric-main-top15").textContent = `${(data.main_top15 * 100).toFixed(1)}%`;
        document.getElementById("metric-euro-top2").textContent = `${(data.euro_top2 * 100).toFixed(1)}%`;
        document.getElementById("metric-euro-top4").textContent = `${(data.euro_top4 * 100).toFixed(1)}%`;
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

// Fetch model predictions & bets
async function fetchPredictions() {
    const listContainer = document.getElementById("bets-list");
    
    try {
        const res = await fetch("/api/predictions");
        const data = await res.json();
        
        if (data.error) {
            listContainer.innerHTML = `
                <div class="loading-spinner-container">
                    <p style="color: #ef4444; font-weight: 500;">AI Model is not trained yet. Please initiate training below.</p>
                </div>
            `;
            // Reset structural ranges
            document.getElementById("val-pred-sum").textContent = "N/A";
            document.getElementById("bar-pred-sum").style.width = "0%";
            return;
        }
        
        predictionsData = data;
        
        // Render structural estimates
        const sum = data.estimated_sum;
        const counts = data.estimated_counts;
        
        document.getElementById("val-pred-sum").textContent = sum.toFixed(2);
        // Map sum (25 to 255) to percentage (0% to 100%)
        const sumPct = Math.min(100, Math.max(0, ((sum - 25) / (255 - 25)) * 100));
        document.getElementById("bar-pred-sum").style.width = `${sumPct}%`;
        
        document.getElementById("val-even-count").textContent = counts.even.toFixed(2);
        document.getElementById("bar-even-count").style.width = `${(counts.even / 5) * 100}%`;
        document.getElementById("val-odd-count").textContent = counts.odd.toFixed(2);
        document.getElementById("bar-odd-count").style.width = `${(counts.odd / 5) * 100}%`;
        
        document.getElementById("val-low-count").textContent = counts.low.toFixed(2);
        document.getElementById("bar-low-count").style.width = `${(counts.low / 5) * 100}%`;
        document.getElementById("val-high-count").textContent = counts.high.toFixed(2);
        document.getElementById("bar-high-count").style.width = `${(counts.high / 5) * 100}%`;
        
        // Render combination bets list
        renderBets();
        
    } catch (err) {
        console.error("Error fetching predictions:", err);
        listContainer.innerHTML = `
            <div class="loading-spinner-container">
                <p style="color: #ef4444; font-weight: 500;">Connection error to quantitative engine.</p>
            </div>
        `;
    }
}

// Renders the prediction combinations in the ticket layout
function renderBets() {
    const listContainer = document.getElementById("bets-list");
    if (!predictionsData) return;
    
    const bets = predictionsData.bets;
    listContainer.innerHTML = "";
    
    bets.forEach((bet) => {
        const mainNums = bet.main_nums;
        const euroNums = bet.euro_nums;
        
        const row = document.createElement("div");
        row.className = "bet-row";
        
        // Label with profile type
        const labelArea = document.createElement("div");
        labelArea.className = "bet-label-area";
        labelArea.style.display = "flex";
        labelArea.style.flexDirection = "column";
        labelArea.style.gap = "2px";
        labelArea.style.minWidth = "110px";
        
        const label = document.createElement("div");
        label.className = "bet-label";
        label.style.fontSize = "13px";
        label.style.fontWeight = "600";
        label.textContent = `Row #${bet.id}`;
        
        const profileBadge = document.createElement("span");
        profileBadge.style.fontSize = "10px";
        profileBadge.style.color = "var(--text-secondary)";
        profileBadge.style.textTransform = "uppercase";
        profileBadge.textContent = bet.profile;
        
        labelArea.appendChild(label);
        labelArea.appendChild(profileBadge);
        row.appendChild(labelArea);
        
        // Balls Group
        const ballsGroup = document.createElement("div");
        ballsGroup.className = "balls-group";
        
        // Main balls
        mainNums.forEach(n => {
            const ball = document.createElement("div");
            ball.className = "ball main";
            ball.innerHTML = `<span>${n}</span>`;
            ballsGroup.appendChild(ball);
        });
        
        // Divider
        const divider = document.createElement("div");
        divider.className = "divider-line";
        ballsGroup.appendChild(divider);
        
        // Euro balls
        euroNums.forEach(e => {
            const ball = document.createElement("div");
            ball.className = "ball euro";
            ball.innerHTML = `<span>${e}</span>`;
            ballsGroup.appendChild(ball);
        });
        
        row.appendChild(ballsGroup);
        
        // Confidence badge
        const confidenceArea = document.createElement("div");
        confidenceArea.className = "confidence-area";
        confidenceArea.style.display = "flex";
        confidenceArea.style.flexDirection = "column";
        confidenceArea.style.alignItems = "flex-end";
        confidenceArea.style.minWidth = "70px";
        
        const confidenceVal = document.createElement("div");
        confidenceVal.style.fontFamily = "var(--font-outfit)";
        confidenceVal.style.fontWeight = "700";
        confidenceVal.style.fontSize = "14px";
        confidenceVal.style.color = bet.confidence > 80 ? "var(--accent-cyan)" : (bet.confidence > 60 ? "var(--accent-violet)" : "var(--accent-gold)");
        confidenceVal.textContent = `${bet.confidence}%`;
        
        const confidenceLabel = document.createElement("span");
        confidenceLabel.style.fontSize = "9px";
        confidenceLabel.style.color = "var(--text-secondary)";
        confidenceLabel.textContent = "Confidence";
        
        confidenceArea.appendChild(confidenceVal);
        confidenceArea.appendChild(confidenceLabel);
        row.appendChild(confidenceArea);
        
        listContainer.appendChild(row);
    });
}

// Trigger Incremental Sync
async function triggerSync() {
    const btn = document.getElementById("btn-sync");
    const text = btn.querySelector(".btn-text");
    const spinner = btn.querySelector(".btn-spinner");
    
    btn.disabled = true;
    text.textContent = "Syncing...";
    spinner.classList.remove("hidden");
    
    try {
        const res = await fetch("/api/update", { method: "POST" });
        const data = await res.json();
        
        if (data.success) {
            if (data.inserted > 0) {
                showToast(`Database updated successfully! Sync'd ${data.inserted} new draws.`);
            } else {
                showToast("Database is already up to date.");
            }
            fetchStatus();
        } else {
            showToast("Failed to synchronize: " + data.error);
        }
    } catch (err) {
        showToast("Network error synchronizing database.");
    } finally {
        btn.disabled = false;
        text.textContent = "Fetch Latest Draw";
        spinner.classList.add("hidden");
    }
}

// Trigger Retrain model
async function triggerTrain() {
    const btn = document.getElementById("btn-train");
    btn.disabled = true;
    
    try {
        const res = await fetch("/api/train", { method: "POST" });
        const data = await res.json();
        
        if (data.success) {
            showToast("Model training initiated in the background.");
            showTrainingProgress(true);
            fetchStatus(); // Will start polling status
        } else {
            showToast("Failed to initiate training: " + data.error);
            btn.disabled = false;
        }
    } catch (err) {
        showToast("Network error initiating training.");
        btn.disabled = false;
    }
}

// Toggle training progress banner
function showTrainingProgress(show) {
    const banner = document.getElementById("training-progress-banner");
    if (show) {
        banner.classList.remove("hidden");
    } else {
        banner.classList.add("hidden");
    }
}

// Helper to show alert/toast messages
function showToast(message) {
    const toast = document.getElementById("toast");
    const msgEl = document.getElementById("toast-message");
    
    msgEl.textContent = message;
    toast.classList.remove("hidden");
    
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 4000);
}
