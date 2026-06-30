// Global state
let currentProfile = 'conservative';
let predictionsData = null;
let pollIntervalId = null;
let activeTicketTab = 'standard';
let currentGame = 'eurojackpot';

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchMetrics();
    fetchPredictions();
    fetchTickets();
    fetchCrypto();
    // Poll crypto prediction every 60 seconds
    setInterval(fetchCrypto, 60000);
});

// Fetch system status
async function fetchStatus() {
    try {
        const res = await fetch(`/api/status?game=${currentGame}`);
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
        if (btnTrain) btnTrain.disabled = (data.training_state === "Training");
        
        // Update register ticket button
        const btnReg = document.getElementById("btn-register");
        if (btnReg) {
            if (data.ticket_registered) {
                btnReg.disabled = true;
                const regType = data.registered_type === 'system' ? 'System Ticket' : 'Standard Ticket';
                btnReg.querySelector(".btn-text").textContent = `Ticket Registered (${regType}) for ${data.next_draw_date}`;
                btnReg.classList.add("secondary");
            } else {
                btnReg.disabled = false;
                const labelType = activeTicketTab === 'system' ? 'System Wheel' : 'This Ticket';
                btnReg.querySelector(".btn-text").textContent = `Register ${labelType} for ${data.next_draw_date}`;
                btnReg.classList.remove("secondary");
            }
        }
        
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
        const res = await fetch(`/api/metrics?game=${currentGame}`);
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
        
        // Render accuracy chart
        if (data.history_hits && data.history_hits.length > 0) {
            const history = data.history_hits.slice(-25); // show last 25 draws
            const labels = history.map(h => {
                const parts = h.date.split("-");
                return parts.length === 3 ? `${parts[1]}-${parts[2]}` : h.date;
            });
            const hitsTop10 = history.map(h => h.hits_top10);
            const hitsTop5 = history.map(h => h.hits_top5);
            
            if (window.myAccuracyChart) {
                window.myAccuracyChart.destroy();
            }
            
            const ctx = document.getElementById('accuracyChart').getContext('2d');
            window.myAccuracyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Hits in Top 10',
                            data: hitsTop10,
                            backgroundColor: 'rgba(6, 182, 212, 0.4)',
                            borderColor: 'rgba(6, 182, 212, 1)',
                            borderWidth: 1.5,
                            borderRadius: 4,
                            barPercentage: 0.6
                        },
                        {
                            label: 'Hits in Top 5 (Single Bet)',
                            data: hitsTop5,
                            type: 'line',
                            borderColor: 'rgba(234, 179, 8, 1)',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            pointBackgroundColor: 'rgba(234, 179, 8, 1)',
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                color: '#94a3b8',
                                font: { family: 'Outfit', size: 10 }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 9 }
                            }
                        },
                        y: {
                            min: 0,
                            max: 5,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                stepSize: 1,
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 9 }
                            }
                        }
                    }
                }
            });
        }
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

// Fetch model predictions & bets
async function fetchPredictions() {
    const listContainer = document.getElementById("bets-list");
    
    try {
        const res = await fetch(`/api/predictions?game=${currentGame}`);
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
    
    const bets = activeTicketTab === 'system' ? predictionsData.system_bets : predictionsData.bets;
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
        
        if (euroNums && euroNums.length > 0) {
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
        }
        
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

// Fetch and render registered tickets
async function fetchTickets() {
    const listContainer = document.getElementById("tickets-tracker-list");
    if (!listContainer) return;
    
    try {
        const res = await fetch(`/api/tickets?game=${currentGame}`);
        const tickets = await res.json();
        
        if (tickets.length === 0) {
            listContainer.innerHTML = `<p style="color: var(--text-secondary); text-align: center; padding: 20px 0; font-size: 13px;">No tickets registered yet.</p>`;
            return;
        }
        
        const groups = {};
        tickets.forEach(t => {
            if (!groups[t.draw_date]) groups[t.draw_date] = [];
            groups[t.draw_date].push(t);
        });
        
        listContainer.innerHTML = "";
        
        Object.keys(groups).forEach(date => {
            const rows = groups[date];
            
            const groupDiv = document.createElement("div");
            groupDiv.className = "ticket-group";
            groupDiv.style.border = "1px solid var(--border-color)";
            groupDiv.style.borderRadius = "10px";
            groupDiv.style.padding = "12px";
            groupDiv.style.background = "rgba(255, 255, 255, 0.01)";
            
            const header = document.createElement("div");
            header.style.display = "flex";
            header.style.justify = "space-between";
            header.style.alignItems = "center";
            header.style.marginBottom = "8px";
            header.style.borderBottom = "1px solid var(--border-color)";
            header.style.paddingBottom = "6px";
            
            const dateLabel = document.createElement("span");
            dateLabel.style.fontSize = "13px";
            dateLabel.style.fontWeight = "600";
            dateLabel.style.fontFamily = "var(--font-outfit)";
            dateLabel.textContent = `Draw Date: ${date}`;
            
            const statusLabel = document.createElement("span");
            statusLabel.style.fontSize = "10px";
            statusLabel.style.padding = "2px 8px";
            statusLabel.style.borderRadius = "10px";
            statusLabel.style.fontWeight = "500";
            
            const isPending = rows.some(r => r.prize_tier === "Pending");
            if (isPending) {
                statusLabel.style.background = "rgba(234, 179, 8, 0.1)";
                statusLabel.style.color = "var(--accent-gold)";
                statusLabel.textContent = "Pending Results";
            } else {
                statusLabel.style.background = "rgba(16, 185, 129, 0.1)";
                statusLabel.style.color = "#10b981";
                statusLabel.textContent = "Evaluated";
            }
            
            header.appendChild(dateLabel);
            header.appendChild(statusLabel);
            groupDiv.appendChild(header);
            
            const rowsList = document.createElement("div");
            rowsList.style.display = "flex";
            rowsList.style.flexDirection = "column";
            rowsList.style.gap = "6px";
            
            rows.forEach(r => {
                const rDiv = document.createElement("div");
                rDiv.style.display = "flex";
                rDiv.style.justify = "space-between";
                rDiv.style.alignItems = "center";
                rDiv.style.fontSize = "11px";
                rDiv.style.color = "var(--text-secondary)";
                
                const meta = document.createElement("span");
                meta.textContent = `Row #${r.row_id} (${r.profile})`;
                rDiv.appendChild(meta);
                
                const mainList = r.main_nums || r.nums || [];
                const euroList = r.euro_nums || [];
                
                const balls = document.createElement("span");
                balls.style.fontFamily = "monospace";
                balls.style.color = "var(--text-primary)";
                if (euroList.length > 0) {
                    balls.textContent = `[${mainList.join(",")}] + [${euroList.join(",")}]`;
                } else {
                    balls.textContent = `[${mainList.join(",")}]`;
                }
                rDiv.appendChild(balls);
                
                const badge = document.createElement("span");
                badge.style.fontSize = "10px";
                badge.style.fontWeight = "600";
                
                if (r.prize_tier === "Pending") {
                    badge.textContent = "Pending";
                    badge.style.color = "var(--text-secondary)";
                } else if (r.prize_tier === "0+0" || r.prize_tier === "0") {
                    badge.textContent = "0";
                    badge.style.color = "var(--text-secondary)";
                } else {
                    if (currentGame === "sportka") {
                        badge.textContent = `${r.prize_tier} tref`;
                    } else {
                        badge.textContent = r.prize_tier;
                    }
                    badge.style.color = "var(--accent-cyan)";
                    badge.style.textShadow = "0 0 5px rgba(6, 182, 212, 0.5)";
                }
                rDiv.appendChild(badge);
                rowsList.appendChild(rDiv);
            });
            
            groupDiv.appendChild(rowsList);
            listContainer.appendChild(groupDiv);
        });
        
    } catch (err) {
        console.error("Error fetching tickets:", err);
    }
}

// Register the current predictions as a ticket for the upcoming draw
async function registerTicket() {
    const btn = document.getElementById("btn-register");
    if (!btn) return;
    
    const text = btn.querySelector(".btn-text");
    const spinner = btn.querySelector(".btn-spinner");
    
    btn.disabled = true;
    text.textContent = "Registering...";
    spinner.classList.remove("hidden");
    
    try {
        const res = await fetch("/api/tickets/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ game: currentGame, type: activeTicketTab })
        });
        const data = await res.json();
        
        if (data.success) {
            showToast(`Ticket successfully registered for draw on ${data.draw_date}!`);
            fetchStatus();
            fetchTickets();
        } else {
            showToast("Failed to register ticket: " + data.error);
            btn.disabled = false;
            text.textContent = activeTicketTab === 'system' ? "Register System Wheel for Next Draw" : "Register This Ticket for Next Draw";
        }
    } catch (err) {
        showToast("Network error registering ticket.");
        btn.disabled = false;
        text.textContent = activeTicketTab === 'system' ? "Register System Wheel for Next Draw" : "Register This Ticket for Next Draw";
    } finally {
        spinner.classList.add("hidden");
    }
}

// Switches between Standard and System ticket tabs
function switchTicketTab(tabType) {
    activeTicketTab = tabType;
    
    const btnStandard = document.getElementById("tab-standard");
    const btnSystem = document.getElementById("tab-system");
    
    if (tabType === 'system') {
        btnSystem.classList.add("active");
        btnSystem.style.color = "var(--accent-cyan)";
        btnSystem.style.borderBottom = "2px solid var(--accent-cyan)";
        btnSystem.style.fontWeight = "600";
        
        btnStandard.classList.remove("active");
        btnStandard.style.color = "var(--text-secondary)";
        btnStandard.style.borderBottom = "none";
        btnStandard.style.fontWeight = "500";
    } else {
        btnStandard.classList.add("active");
        btnStandard.style.color = "var(--accent-cyan)";
        btnStandard.style.borderBottom = "2px solid var(--accent-cyan)";
        btnStandard.style.fontWeight = "600";
        
        btnSystem.classList.remove("active");
        btnSystem.style.color = "var(--text-secondary)";
        btnSystem.style.borderBottom = "none";
        btnSystem.style.fontWeight = "500";
    }
    
    renderBets();
    fetchStatus();
}

function switchGame(gameType) {
    currentGame = gameType;
    
    const btnEuro = document.getElementById("game-eurojackpot");
    const btnSportka = document.getElementById("game-sportka");
    const btnRentier = document.getElementById("game-rentier");
    
    const lotteryLayout = document.getElementById("lottery-layout");
    const rentierLayout = document.getElementById("rentier-layout");
    
    // Style reset helper
    function setButtonActive(btn, active) {
        if (active) {
            btn.style.background = "rgba(0, 242, 254, 0.1)";
            btn.style.borderColor = "var(--accent-cyan)";
            btn.style.color = "var(--accent-cyan)";
        } else {
            btn.style.background = "rgba(255, 255, 255, 0.03)";
            btn.style.borderColor = "var(--border-color)";
            btn.style.color = "var(--text-secondary)";
        }
    }
    
    if (gameType === 'rentier') {
        setButtonActive(btnRentier, true);
        setButtonActive(btnEuro, false);
        setButtonActive(btnSportka, false);
        
        lotteryLayout.classList.add("hidden");
        lotteryLayout.style.display = "none";
        rentierLayout.classList.remove("hidden");
        rentierLayout.style.display = "flex";
        
        // Update header & subtitle
        document.querySelector(".title-meta h1").innerHTML = 'Rentierský Plán <span class="accent-text">(10 let)</span>';
        document.querySelector(".title-meta p").textContent = 'AI Trading Portfolio Savings & Dynamic Deleveraging Tracker';
        
        fetchPlan();
        return;
    }
    
    // For standard games, ensure lottery layout is visible
    lotteryLayout.classList.remove("hidden");
    lotteryLayout.style.display = "flex";
    rentierLayout.classList.add("hidden");
    rentierLayout.style.display = "none";
    
    setButtonActive(btnRentier, false);
    
    if (gameType === 'sportka') {
        setButtonActive(btnSportka, true);
        setButtonActive(btnEuro, false);
        
        // Hide Euro metrics rows
        document.getElementById("row-metric-euro2").style.display = "none";
        document.getElementById("row-metric-euro4").style.display = "none";
        
        // Update header & subtitle
        document.querySelector(".title-meta h1").innerHTML = 'Sportka <span class="accent-text">AI Forecast</span>';
        document.querySelector(".title-meta p").textContent = 'Self-tuning LSTM neural network & magnitude pruning (6 main + supplementary)';
    } else {
        setButtonActive(btnEuro, true);
        setButtonActive(btnSportka, false);
        
        // Show Euro metrics rows
        document.getElementById("row-metric-euro2").style.display = "flex";
        document.getElementById("row-metric-euro4").style.display = "flex";
        
        document.querySelector(".title-meta h1").innerHTML = 'Eurojackpot <span class="accent-text">AI Forecast</span>';
        document.querySelector(".title-meta p").textContent = 'Self-tuning LSTM neural network & magnitude pruning';
    }
    
    document.getElementById("tab-system").textContent = gameType === 'sportka' ? 'System Wheel (8 numbers)' : 'System Wheel (7 numbers)';
    
    // Clear prediction container loader before fetch
    document.getElementById("bets-list").innerHTML = `
        <div class="loading-spinner-container">
            <div class="spinner"></div>
            <p>Generating neural predictions...</p>
        </div>
    `;
    
    // Refresh all data
    fetchStatus();
    fetchMetrics();
    fetchPredictions();
    fetchTickets();
    fetchQuantum();
}

async function fetchPlan() {
    const tableBody = document.getElementById("plan-table-body");
    tableBody.innerHTML = `
        <tr>
            <td colspan="7" style="text-align: center; padding: 30px;">
                <div class="spinner" style="margin: 0 auto;"></div>
                <p style="margin-top: 10px; color: var(--text-secondary);">Načítám 10letý finanční plán...</p>
            </td>
        </tr>
    `;
    
    try {
        const res = await fetch("/api/crypto/plan");
        const data = await res.json();
        
        if (data.error) {
            tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444; padding: 20px;">Error: ${data.error}</td></tr>`;
            return;
        }
        
        tableBody.innerHTML = "";
        data.forEach(item => {
            const tr = document.createElement("tr");
            tr.style.borderBottom = "1px solid var(--border-color)";
            
            // Deviation math
            let devHtml = `<span style="color: var(--text-secondary);">—</span>`;
            if (item.actual_balance !== null) {
                const dev = item.actual_balance - item.expected_balance;
                const devFormatted = dev.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                if (dev >= 0) {
                    devHtml = `<span style="color: var(--accent-cyan); font-weight: 600;">+${devFormatted} USDT</span>`;
                } else {
                    devHtml = `<span style="color: #ef4444; font-weight: 600;">${devFormatted} USDT</span>`;
                }
            }
            
            tr.innerHTML = `
                <td style="padding: 12px; font-weight: 500;">${item.month}</td>
                <td style="padding: 12px; color: var(--text-secondary);">${item.deposit.toFixed(1)} USDT</td>
                <td style="padding: 12px; color: var(--text-secondary);">${item.leverage}x / ${item.alloc_pct}%</td>
                <td style="padding: 12px; color: var(--accent-cyan); font-weight: 500;">+${item.rate_pct}%</td>
                <td style="padding: 12px; font-family: monospace; font-weight: 600;">${item.expected_balance.toLocaleString()} USDT</td>
                <td style="padding: 12px;">
                    <input type="number" step="0.01" value="${item.actual_balance !== null ? item.actual_balance : ''}" 
                           placeholder="—" 
                           onchange="updateActualBalance(${item.index}, this.value)" 
                           style="width: 110px; background: rgba(0,0,0,0.25); border: 1px solid var(--border-color); color: var(--text-primary); border-radius: 4px; padding: 4px 8px; text-align: right; font-family: monospace; outline: none; transition: border-color 0.2s;">
                </td>
                <td style="padding: 12px; font-family: monospace;">${devHtml}</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (err) {
        console.error("Error fetching plan:", err);
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444; padding: 20px;">Nepodařilo se připojit k serveru.</td></tr>`;
    }
}

async function updateActualBalance(index, value) {
    const valFloat = value.trim() === "" ? null : parseFloat(value);
    
    try {
        const res = await fetch("/api/crypto/plan/update", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                index: index,
                actual_balance: valFloat
            })
        });
        const data = await res.json();
        
        if (data.success) {
            showToast("Zůstatek úspěšně uložen!");
            // Re-fetch plan to update the deviation column immediately
            fetchPlan();
        } else {
            showToast("Chyba při ukládání: " + data.error);
        }
    } catch (err) {
        console.error("Error saving actual balance:", err);
        showToast("Nepodařilo se uložit zůstatek.");
    }
}

async function fetchCrypto() {
    try {
        const res = await fetch("/api/crypto/live");
        const data = await res.json();
        
        if (data.error) {
            ["btc", "eth", "sol"].forEach(coin => {
                document.getElementById(`${coin}-price`).textContent = "N/A";
                document.getElementById(`${coin}-prediction`).textContent = "N/A";
            });
            return;
        }
        
        const coins = ["btcusdt", "ethusdt", "solusdt"];
        coins.forEach(coinKey => {
            const coinData = data[coinKey];
            if (!coinData) return;
            
            const coinPrefix = coinKey.replace("usdt", "");
            
            // Update price
            document.getElementById(`${coinPrefix}-price`).textContent = `${coinData.price.toLocaleString()} USDT`;
            
            // Update prediction with styling
            const predEl = document.getElementById(`${coinPrefix}-prediction`);
            predEl.textContent = coinData.prediction;
            if (coinData.prediction === "UP") {
                predEl.style.color = "var(--accent-cyan)";
            } else {
                predEl.style.color = "#ef4444";
            }
            
            // Update indicators
            document.getElementById(`${coinPrefix}-confidence`).textContent = `${coinData.probability.toFixed(1)}%`;
            document.getElementById(`${coinPrefix}-winrate`).textContent = `${coinData.win_rate.toFixed(1)}%`;
            
            // Update Action with styling
            const actionEl = document.getElementById(`${coinPrefix}-action`);
            actionEl.textContent = coinData.action;
            if (coinData.action === "BUY / LONG") {
                actionEl.style.color = "var(--accent-cyan)";
            } else if (coinData.action === "SELL / SHORT") {
                actionEl.style.color = "#ef4444";
            } else {
                actionEl.style.color = "var(--text-secondary)";
            }
            
            // Update SL and TP
            document.getElementById(`${coinPrefix}-sl`).textContent = coinData.stop_loss === "N/A" ? "N/A" : `${coinData.stop_loss.toLocaleString()} USDT`;
            document.getElementById(`${coinPrefix}-tp`).textContent = coinData.take_profit === "N/A" ? "N/A" : `${coinData.take_profit.toLocaleString()} USDT`;
        });
        
        // Format last updated datetime
        const dateParts = data.updated_at.split(" ");
        document.getElementById("crypto-updated").textContent = dateParts.length === 2 ? `${dateParts[0]} ${dateParts[1]}` : data.updated_at;
        
    } catch (err) {
        console.error("Error fetching crypto metrics:", err);
    }
}

async function fetchQuantum() {
    const container = document.getElementById("quantum-numbers-display");
    if (!container) return;
    container.innerHTML = `<span style="color: var(--text-secondary); font-size: 13px;">Performing quantum measurement...</span>`;
    
    try {
        const res = await fetch(`/api/lotto/quantum?game=${currentGame}`);
        const data = await res.json();
        
        if (data.error) {
            container.innerHTML = `<span style="color: #ef4444; font-size: 12px;">Error: ${data.error}</span>`;
            return;
        }
        
        container.innerHTML = "";
        // Render main numbers
        data.quantum_main.forEach(num => {
            const ball = document.createElement("span");
            ball.className = "ball";
            ball.textContent = num;
            ball.style.width = "30px";
            ball.style.height = "30px";
            ball.style.lineHeight = "30px";
            ball.style.fontSize = "12px";
            ball.style.fontWeight = "600";
            ball.style.display = "inline-block";
            ball.style.textAlign = "center";
            ball.style.borderRadius = "50%";
            ball.style.background = "linear-gradient(135deg, var(--accent-cyan), #0072ff)";
            ball.style.color = "#000";
            ball.style.boxShadow = "0 0 10px rgba(0, 242, 254, 0.4)";
            ball.style.margin = "0 2px";
            container.appendChild(ball);
        });
        
        // Add a divider
        if (data.quantum_euro.length > 0) {
            const divider = document.createElement("span");
            divider.textContent = "|";
            divider.style.color = "var(--border-color)";
            divider.style.margin = "0 6px";
            divider.style.fontWeight = "bold";
            container.appendChild(divider);
        }
        
        // Render euro/supplementary numbers
        data.quantum_euro.forEach(num => {
            const ball = document.createElement("span");
            ball.className = "ball euro";
            ball.textContent = num;
            ball.style.width = "30px";
            ball.style.height = "30px";
            ball.style.lineHeight = "30px";
            ball.style.fontSize = "12px";
            ball.style.fontWeight = "600";
            ball.style.display = "inline-block";
            ball.style.textAlign = "center";
            ball.style.borderRadius = "50%";
            ball.style.background = "linear-gradient(135deg, #ffe259, #ffa751)";
            ball.style.color = "#000";
            ball.style.boxShadow = "0 0 10px rgba(255, 226, 89, 0.4)";
            ball.style.margin = "0 2px";
            container.appendChild(ball);
        });
        
        document.getElementById("quantum-purity").textContent = data.qrc_energy.toFixed(4);
        
    } catch (err) {
        console.error("Error fetching quantum prediction:", err);
        container.innerHTML = `<span style="color: #ef4444; font-size: 12px;">Failed to collapse state.</span>`;
    }
}
