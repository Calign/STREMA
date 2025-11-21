// =====================
// script.js (Full) - Integrated for Login, Detection, Result, History, Recommendation
// =====================

// --------------------
// Login / Signup / Logout
// --------------------
async function signup() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const result = await eel.signup(username, password)();
    document.getElementById("result").innerText = result;
}

async function login() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const result = await eel.login(username, password)();
    document.getElementById("result").innerText = result.msg || result;

    if (result.status === "success") {
        localStorage.setItem("username", username);
        setTimeout(() => window.location.href = "home.html", 800);
    }
}

// --------------------
// Logout System (Custom Modal)
// --------------------

function logoutUser() {
    // Show the modal instead of using confirm()
    const modal = document.getElementById("logoutModal");
    modal.classList.add("show");
}

function closeLogoutModal() {
    const modal = document.getElementById("logoutModal");
    modal.classList.remove("show");
}

async function confirmLogout() {
    // Proceed with actual logout
    localStorage.clear();
    sessionStorage.clear();

    try { 
        await eel.logout()(); 
    } catch(e) {
        console.warn("Logout failed:", e);
    }

    window.location.href = "login.html";
}


function navigateWithSlide(targetPage, direction) {
    document.body.classList.add(direction === "left" ? "slide-out-left" : "slide-out-right");
    setTimeout(() => window.location.href = targetPage, 400);
}

// -------- Display Username on Pages ---------- //
document.addEventListener("DOMContentLoaded", () => {
    const el = document.getElementById("displayUsername");
    if (el) {
        const username = localStorage.getItem("username") || "User";
        el.textContent = username;
    }
});



if (window.location.pathname.includes("home.html")) {
    document.addEventListener("DOMContentLoaded", async () => {
        try {
            // Fetch last stress value from backend (single number)
            const lastStress = await eel.get_last_result()();
            console.log("Last session stress:", lastStress);

            const stressTextEl = document.getElementById("recentStressText");
            const spriteImg = document.getElementById("stressSprite");

            if (stressTextEl) stressTextEl.textContent = lastStress != null ? lastStress : "N/A";

            if (spriteImg) {
                if (lastStress >= 1 && lastStress <= 4) spriteImg.src = "assets/sprite/sprite1.png";
                else if (lastStress >= 5 && lastStress <= 7) spriteImg.src = "assets/sprite/sprite2.png";
                else if (lastStress >= 8 && lastStress <= 10) spriteImg.src = "assets/sprite/sprite3.png";
                else spriteImg.src = "";
            }

            // Load last 5 sessions for chart
            const chartEl = document.getElementById("homeStressChart");
            if (chartEl) {
                const username = localStorage.getItem("username") || "Guest";
                

                try {
                    const history = await eel.get_full_detection_history(username)() || [];

                    // Take last 5 sessions (oldest → newest)
                    const last5 = history.slice(-5);
                    const labels = last5.map(item => `${item.date ?? "N/A"} ${item.time ?? ""}`);
                    const stressValues = last5.map(item => Number(item.stress_level) || 0);

                    console.log("Chart labels:", labels);
                    console.log("Chart values:", stressValues);

                    const ctx = chartEl.getContext("2d");
                    if (window.homeChartInstance) window.homeChartInstance.destroy();

                    window.homeChartInstance = new Chart(ctx, {
                        type: "line",
                        data: {
                            labels,
                            datasets: [{
                                label: "Stress Level",
                                data: stressValues,
                                borderColor: "#3F9334",
                                backgroundColor: "rgba(63, 147, 52, 0.15)",
                                borderWidth: 3,
                                pointBackgroundColor: "#3F9334",
                                pointRadius: 6,
                                tension: 0.35,
                                fill: true,
                            }]
                        },
                        options: {
                            scales: {
                                y: { beginAtZero: true, max: 10 },
                                x: { ticks: { autoSkip: false } }
                            },
                            plugins: { legend: { display: false } },
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });

                } catch (e) {
                    console.error("Failed to load last 5 sessions chart:", e);
                }
            }

        } catch (e) {
            console.error("Failed to load home page data:", e);
        }
    });
}









// --------------------
// Detection Page Logic (Fixed)
// --------------------
if (window.location.pathname.includes("detection.html")) {
    const videoEl = document.getElementById("webcam");
    const startBtn = document.getElementById("startBtn");
    const cancelBtn = document.getElementById("cancelBtn");
    const timerDisplay = document.getElementById("timerDisplay");
    const modeSelect = document.getElementById("detection-mode");
    const currentHR = document.getElementById("currentHR");
    const overlay = document.getElementById("overlayCanvas");
    const overlayCtx = overlay.getContext("2d");

    let stream = null;
    let hrSeries = [];
    let detectTimer = null;
    let frameCaptureInterval = null;
    let hrPollInterval = null;
    let running = false;
    const detectionDuration = 10; // seconds
    const frameCaptureMs = 500; // capture every 500ms
    let animationId = null;

    async function startWebcam() {
        if (stream) return true;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoEl.srcObject = stream;
            await videoEl.play();
            return true;
        } catch (err) {
            alert("Cannot access webcam.");
            console.error(err);
            return false;
        }
    }

    function stopWebcam() {
        if (!stream) return;
        stream.getTracks().forEach(t => t.stop());
        stream = null;
        videoEl.srcObject = null;
    }

    function captureFrameDataUrl() {
        if (!videoEl.videoWidth || !videoEl.videoHeight) return null;
        const canvas = document.createElement("canvas");
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/png");
    }

    function formatTime(sec) {
        const m = Math.floor(sec / 60).toString().padStart(2, "0");
        const s = Math.floor(sec % 60).toString().padStart(2, "0");
        return `${m}:${s}`;
    }

    let lastHRValue = null;

    // Animate last HR values from CSV at start (optional)
    function animateHeartRates(hrArray, intervalMs = 1500) {
        if (!hrArray || hrArray.length === 0) return;
        let index = 0;

        function next() {
            if (index >= hrArray.length) return;
            if (!currentHR.dataset.locked) {
                lastHRValue = Number(hrArray[index]); // update lastHRValue
                currentHR.innerText = lastHRValue;
                currentHR.classList.add("latest");
                currentHR.dataset.locked = "true";

                setTimeout(() => {
                    currentHR.classList.remove("latest");
                    delete currentHR.dataset.locked;
                }, intervalMs * 0.8);
            }
            index++;
            setTimeout(next, intervalMs);
        }
        next();
    }

    // Poll the latest HR from Python continuously
    async function pollLatestHR() {
        try {
            const hrVal = await eel.get_latest_heart_rate_value()();
            if (hrVal !== undefined && hrVal !== null) {
                lastHRValue = Number(hrVal);
                hrSeries.push(lastHRValue);
                if (!currentHR.dataset.locked) {
                    currentHR.innerText = lastHRValue;
                    currentHR.classList.add("latest");
                    setTimeout(() => currentHR.classList.remove("latest"), 1200);
                }
            } else {
                // only show -- if we never had a valid value
                if (lastHRValue === null) currentHR.innerText = "--";
            }
        } catch (e) {
            console.error("HR polling failed:", e);
            // keep lastHRValue if we had a previous reading
            if (lastHRValue !== null) currentHR.innerText = lastHRValue;
            else currentHR.innerText = "--";
        }
    }



    function resizeOverlay() {
        overlay.width = videoEl.videoWidth;
        overlay.height = videoEl.videoHeight;
    }

    function drawFeatures(featurePoints = [], hr = null) {
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

        if (!featurePoints || featurePoints.length === 0) return;

        featurePoints.forEach(pt => {
            overlayCtx.beginPath();
            overlayCtx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
            overlayCtx.fillStyle = "lime";
            overlayCtx.fill();
            overlayCtx.strokeStyle = "cyan";
            overlayCtx.lineWidth = 1;
            overlayCtx.stroke();
        });
    }


    async function onDetectionComplete(frames) {
        running = false;
        cancelBtn.style.display = "none";
        startBtn.style.display = "inline-block";
        stopWebcam();

        if (animationId) cancelAnimationFrame(animationId);

        // Show loading overlay
        const overlayEl = document.getElementById("loadingOverlay");
        overlayEl.style.display = "flex";
        overlayEl.classList.add("show");

        const mode = modeSelect.value;
        const username = localStorage.getItem("username") || "Guest";
        let detectRes = null;

        try {
            detectRes = await eel.run_detection(frames, mode, username)();
        } catch (e) {
            console.error("run_detection failed:", e);
        }

        localStorage.setItem("last_detection_result", JSON.stringify({
            detected: detectRes,
            hr_series: hrSeries,
            captured_frame: frames.length > 0 ? frames[frames.length - 1] : null
        }));

        if (detectRes) {
            const payload = {
                username,
                timestamp: new Date().toISOString(),
                mode: detectRes.mode || mode,
                face_present: detectRes.face_present || false,
                frames_used: frames.length,
                cnn_label: detectRes.cnn_label ?? null,
                cnn_confidence: detectRes.cnn_confidence ?? null,
                hr_value: detectRes.hr_value ?? null,
                xgb_pred: detectRes.xgb_pred ?? null,
                combined: detectRes.combined ?? null
            };
            try { await eel.save_detection_result(payload)(); }
            catch (e) { console.error("Save failed:", e); }
        }

        // Wait a tiny bit so spinner is visible even if processing is super fast
        await new Promise(res => setTimeout(res, 200));

        // Hide overlay (optional) and redirect
        // overlayEl.style.display = "none";
        window.location.href = "result.html";
    }


    async function startDetection() {
        try {
            console.log("Detection started");
            if (running) return;
            running = true;
            startBtn.style.display = "none";
            cancelBtn.style.display = "inline-block";
            hrSeries = [];

            const webcamStarted = await startWebcam();
            if (!webcamStarted) {
                running = false;
                startBtn.style.display = "inline-block";
                cancelBtn.style.display = "none";
                return;
            }

            // Wait for video metadata
            await new Promise(resolve => {
                if (videoEl.videoWidth && videoEl.videoHeight) resolve();
                else videoEl.onloadedmetadata = () => resolve();
            });

            resizeOverlay();

            // Start feature overlay animation
            animationId = requestAnimationFrame(async function renderLoop() {
                if (!running) return;
                try {
                    const frameData = captureFrameDataUrl();
                    if (frameData) {
                        // Fetch all 26 feature points from Python
                        const features = await eel.get_current_face_feature_points(frameData)();
                        drawFeatures(features);
                    }
                } catch (e) { console.warn("Feature loop failed:", e); }
                animationId = requestAnimationFrame(renderLoop);
            });


            // Sync HR
            try { await eel.sync_heart_rate()(); } catch (e) { console.warn("Auto-sync failed:", e); }

            // Animate last 10 HR
            try {
                const lastHR = await eel.get_last_n_heart_rates(10)();
                animateHeartRates(lastHR, 1500);
            } catch (e) { console.warn("Failed HR animation:", e); }

            await pollLatestHR();
            hrPollInterval = setInterval(pollLatestHR, 1000);

            // Capture frames
            const frames = [];
            frameCaptureInterval = setInterval(() => {
                const durl = captureFrameDataUrl();
                if (!durl) return;
                frames.push(durl);
            }, frameCaptureMs);

            // Countdown timer
            let remaining = detectionDuration;
            timerDisplay.textContent = formatTime(remaining);
            detectTimer = setInterval(async () => {
                remaining--;
                timerDisplay.textContent = formatTime(remaining);
                if (remaining <= 0) {
                    clearInterval(detectTimer);
                    clearInterval(frameCaptureInterval);
                    clearInterval(hrPollInterval);
                    await onDetectionComplete(frames);
                }
            }, 1000);

        } catch (e) {
            console.error("startDetection failed:", e);
            running = false;
            startBtn.style.display = "inline-block";
            cancelBtn.style.display = "none";
        }
    }

    function cancelDetection() {
        if (!running) return;
        running = false;
        clearInterval(detectTimer);
        clearInterval(frameCaptureInterval);
        clearInterval(hrPollInterval);
        timerDisplay.textContent = "00:00";
        startBtn.style.display = "inline-block";
        cancelBtn.style.display = "none";
        stopWebcam();
        currentHR.innerText = "--";
        if (animationId) cancelAnimationFrame(animationId);
    }

    startBtn.addEventListener("click", startDetection);
    cancelBtn.addEventListener("click", () => { if (confirm("Stop detection?")) cancelDetection(); });

    startWebcam().catch(() => { });
}



// --------------------
// Result Page Logic
// --------------------
if(window.location.pathname.includes("result.html")){
    const stressEl = document.getElementById("stressLevel");
    const modeEl = document.getElementById("mode");
    const cnnEl = document.getElementById("cnnResult");
    const hrEl = document.getElementById("hrValue");
    const xgbEl = document.getElementById("xgbResult");
    const faceEl = document.getElementById("faceDetected");
    const imgEl = document.getElementById("capturedImg");
    const recEl = document.getElementById("stressRecommendation");
    const confEl = document.getElementById("cnnConf"); // Added missing element
    const scaleInfoEl = document.getElementById("stressScaleInfo"); // Added missing element

    const stored = localStorage.getItem("last_detection_result");

    if(stored){
        const {detected, captured_frame} = JSON.parse(stored);
        const fmt = (v, d=1) => v != null ? Number(v).toFixed(d) : "--";

        // Display detection results
        stressEl.innerText = detected.combined != null ? `${fmt(detected.combined)} / 10` : "--";
        modeEl.innerText = detected.mode || "both";
        cnnEl.innerText = detected.cnn_label != null ? fmt(detected.cnn_label) : "N/A";
        xgbEl.innerText = detected.xgb_pred != null ? fmt(detected.xgb_pred) : "N/A";
        hrEl.innerText = detected.hr_value != null ? `${detected.hr_value} bpm` : "N/A";
        confEl.innerText = detected.cnn_confidence != null ? detected.cnn_confidence.toFixed(3) : "N/A";
        faceEl.innerText = detected.face_present ? "Yes" : "No";
        if(captured_frame && imgEl) imgEl.src = captured_frame;

            // === Stress Bar Indicator ===
        const indicator = document.getElementById("stress-indicator");
        if (indicator && detected.combined != null) {
            const score = Math.min(Math.max(Number(detected.combined), 1), 10); // clamp 1-10
            const position = ((score - 1) / 9) * 100; // map 1→0%, 10→100%
            indicator.style.left = `calc(${position}% - 10px)`; // offset for centering
    }


        // Set stress scale description dynamically
        if(detected.combined != null){
            const combinedScore = Number(detected.combined);
            if(combinedScore >= 1 && combinedScore <= 4){
                scaleInfoEl.innerText = "Little Stress.";
            } else if(combinedScore >= 5 && combinedScore <= 7){
                scaleInfoEl.innerText = "Medium Stress.";
            } else if(combinedScore >= 8){
                scaleInfoEl.innerText = "High Stress.";
            } else {
                scaleInfoEl.innerText = "Stress level unavailable.";
            }
        } else {
            scaleInfoEl.innerText = "Stress level unavailable.";
        }

        // Load Recommendations
        if (detected && detected.combined != null) {
            eel.get_recommendation_for_stress(detected.combined)()
                .then(res => {
                    console.log("DEBUG: Recommendation received from Python:", res);
                    const recs = res.recommendations || [];
                    recEl.innerHTML = recs.length > 0
                        ? `<ul>${recs.map(r => `<li>${r}</li>`).join("")}</ul>`
                        : "<p>No recommendations available.</p>";
                })
                .catch(err => {
                    console.error("Failed to fetch recommendation:", err);
                    recEl.innerHTML = "<p>No recommendations available.</p>";
                });
        } else {
            recEl.innerHTML = "<p>No recommendations available.</p>";
        }

    } else {
        stressEl.innerText = "No detection data found.";
        scaleInfoEl.innerText = "";
        recEl.innerHTML = "<p>No recommendations available.</p>";
    }


    // Fetch last scaled facial features from backend
    eel.get_last_scaled_facial_features()().then(featureArrays => {
        if (!featureArrays || featureArrays.length === 0) return;

        // featureArrays is list of arrays per frame
        const labels = [
            "SAu01_InnerBrowRaiser",   // left eyebrow height
            "SAu02_OuterBrowRaiser",   // right eyebrow height
            "SAu04_BrowLowerer",       // left eyebrow length / brow distance
            "SAu05_UpperLidRaiser",    // upper lip height
            "SAu06_CheekRaiser",       // left eye height
            "SAu07_LidTightener",      // face width ratio
            "SAu09_NoseWrinkler",      // nose wrinkle
            "SAu10_UpperLipRaiser",    // other feature / lip raiser
            "SAu12_LipCornerPuller",   // left mouth upper dist
            "SAu14_Dimpler",            // right mouth upper dist
            "SAu15_LipCornerDepressor", // left eyebrow dist
            "SAu17_ChinRaiser",         // chin-nose distance
            "SAu20_LipStretcher",       // left mouth - right mouth dist
            "SAu23_LipTightener",       // right eyebrow dist
            "SAu24_LipPressor",         // left mouth - lower lip
            "SAu25_LipsPart",           // upper lip - lower lip
            "SAu26_JawDrop",            // right mouth - lower lip
            "SAu27_MouthStretch",       // upper lip - lower lip (duplicate)
            "SAu43_EyesClosed",         // left eye open fraction
            "SmouthOpen",               // right eye open fraction
            "SleftEyeClosed",           // left eye width
            "SrightEyeClosed",          // right eye width
            "SleftEyebrowLowered",      // left eyebrow to eye gap
            "SleftEyebrowRaised",       // left eye to eyebrow gap
            "SrightEyebrowLowered",     // right eyebrow to eye gap
            "SrightEyebrowRaised"       // right eye to eyebrow gap
        ];


        // Average features across frames
        const avgFeatures = labels.map((_, i) => {
            let sum = 0;
            featureArrays.forEach(f => { sum += f[i]; });
            return sum / featureArrays.length;
        });

        const ctx = document.getElementById("featureGraph").getContext("2d");
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Intensity',
                    data: avgFeatures,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Facial Feature Intensities (averaged per frame)'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    });

}


// --------------------
// Enhanced History Page Logic
// --------------------
if (window.location.pathname.includes("history.html")) {
    const tbody = document.getElementById("historyBody");
    const chartCanvas = document.getElementById("stressChart");
    const summaryEl = document.getElementById("historySummary");
    const filterByTime = document.getElementById("filterByTime");
    const filterByMode = document.getElementById("filterByMode");

    async function loadFullData() {
        const username = localStorage.getItem("username") || "Guest";
        const data = await eel.get_full_detection_history(username)();
        return data || [];
    }

    function filterData(data, timeRange, mode) {
        const now = new Date();
        return data.filter(entry => {
            const entryDate = new Date(entry.date + " " + entry.time);
            const sameMode =
                mode === "all" || (entry.mode && entry.mode.toLowerCase() === mode.toLowerCase());
            if (!sameMode) return false;

            switch (timeRange) {
                case "day":
                    return entryDate.toDateString() === now.toDateString();
                case "week": {
                    const firstDay = new Date(now);
                    firstDay.setDate(now.getDate() - now.getDay());
                    firstDay.setHours(0,0,0,0);
                    const lastDay = new Date(firstDay);
                    lastDay.setDate(firstDay.getDate() + 7);
                    return entryDate >= firstDay && entryDate < lastDay;
                }
                case "month":
                    return entryDate.getMonth() === now.getMonth() && entryDate.getFullYear() === now.getFullYear();
                case "year":
                    return entryDate.getFullYear() === now.getFullYear();
                default:
                    return true;
            }
        });
    }

    function aggregateData(data, timeRange) {
        if (!data.length) return [];

        if (timeRange === "day") {
            return data
                .sort((a,b) => new Date(a.date+" "+a.time) - new Date(b.date+" "+b.time)) // oldest → newest
                .map(d => ({ label: d.time, stress_avg: d.stress_level }));
        }

        const grouped = {};
        data.forEach(d => {
            const date = new Date(d.date + " " + d.time);
            let label;
            if (timeRange === "week") label = date.toLocaleDateString("en-US", { weekday: "short" });
            else if (timeRange === "month") label = date.getDate().toString();
            else if (timeRange === "year") label = date.toLocaleString("en-US", { month: "short" });
            else label = d.date;

            if (!grouped[label]) grouped[label] = [];
            if (d.stress_level != null) grouped[label].push(d.stress_level);
        });

        return Object.entries(grouped).map(([label, vals]) => ({
            label,
            stress_avg: vals.reduce((a, b) => a + b, 0) / vals.length,
        }))
        .sort((a,b) => a.label.localeCompare(b.label)); // ensure chronological
    }

    async function loadHistoryAndChart() {
        const rawData = await loadFullData();
        const timeRange = filterByTime.value;
        const mode = filterByMode.value;

        const filtered = filterData(rawData, timeRange, mode);
        const aggregated = aggregateData(filtered, timeRange);

        tbody.innerHTML = "";
        summaryEl.innerHTML = "";

        if (!filtered.length) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;">No records found.</td></tr>`;
            summaryEl.innerHTML = `<p>No detection records for this filter.</p>`;
            if (window.stressChartInstance) window.stressChartInstance.destroy();
            return;
        }

        filtered.sort((a,b)=>new Date(a.date+" "+a.time) - new Date(b.date+" "+b.time)); // oldest → newest

        filtered.forEach((entry, idx)=>{
            const tr=document.createElement("tr");
            tr.innerHTML=`
                <td>${idx+1}</td>
                <td>${entry.date||"--"}</td>
                <td>${entry.time||"--"}</td>
                <td>${entry.mode||"--"}</td>
                <td>${entry.hr_value!=null?entry.hr_value+" bpm":"--"}</td>
                <td>${entry.stress_level!=null?entry.stress_level.toFixed(1):"--"}</td>
            `;
            tbody.appendChild(tr);
        });

        const avgStress = (filtered.reduce((a,e)=>a+(e.stress_level||0),0)/filtered.length).toFixed(2);
        const avgHR = (filtered.reduce((a,e)=>a+(e.hr_value||0),0)/filtered.length).toFixed(2);
        summaryEl.innerHTML = `
            <div class="summary-card">
                <p><b>Total Records:</b> ${filtered.length}</p>
                <p><b>Average Stress:</b> ${avgStress}</p>
                <p><b>Average Heart Rate:</b> ${avgHR} bpm</p>
            </div>
        `;

        if (window.stressChartInstance) window.stressChartInstance.destroy();
        const ctx = chartCanvas.getContext("2d");
        window.stressChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: aggregated.map(d=>d.label),  // oldest → newest
                datasets: [{
                    label:"Stress Level",
                    data: aggregated.map(d=>d.stress_avg),
                    fill:true,
                    backgroundColor:"rgba(75,192,192,0.2)",
                    borderColor:"rgba(75,192,192,1)",
                    tension:0.3,
                    pointRadius:5,
                    pointHoverRadius:7
                }]
            },
            options:{
                responsive:true,
                maintainAspectRatio:false,
                plugins:{legend:{display:true,position:"top"},tooltip:{mode:"index",intersect:false}},
                scales:{y:{min:0,max:10,ticks:{stepSize:1}},x:{ticks:{autoSkip:false}}}
            }
        });
    }

    filterByTime.addEventListener("change", loadHistoryAndChart);
    filterByMode.addEventListener("change", loadHistoryAndChart);

    loadHistoryAndChart();
}




// --------------------
// Breathing Exercise (Guided with Hold Phases)
// --------------------
const startBreathingBtn = document.getElementById("startBreathingBtn");
const stopBreathingBtn = document.getElementById("stopBreathingBtn");
const breathingContainer = document.getElementById("breathingContainer");
const breathingCircle = document.getElementById("breathingCircle");
const breathingText = document.getElementById("breathingText");
const timerDisplay = document.getElementById("timerDisplay");

let breathingInterval;
let totalTime = 60; // total exercise time in seconds
let breathingPhases = [
    { text: "Breathe in", note: "Please breathe in through your nose", duration: 4 },
    { text: "Hold", note: "Hold your breath gently", duration: 4 },
    { text: "Breathe out", note: "Please breathe out through your mouth", duration: 4 },
    { text: "Hold", note: "Relax", duration: 4 }
];

startBreathingBtn.addEventListener("click", () => {
    breathingContainer.classList.remove("hidden");
    let phaseIndex = 0;
    let phaseTime = breathingPhases[phaseIndex].duration;
    let remainingTime = totalTime;

    updateBreathingText();

    breathingInterval = setInterval(() => {
        remainingTime--;
        phaseTime--;
        timerDisplay.textContent = formatTime(remainingTime);

        if (phaseTime <= 0) {
            phaseIndex = (phaseIndex + 1) % breathingPhases.length;
            phaseTime = breathingPhases[phaseIndex].duration;
            updateBreathingText();
        }

        if (remainingTime <= 0) {
            clearInterval(breathingInterval);
            breathingText.textContent = "Well done!";
            breathingCircle.style.transform = "scale(1)";
        }
    }, 1000);

    function updateBreathingText() {
        const phase = breathingPhases[phaseIndex];
        breathingText.textContent = `${phase.text}\n${phase.note}`;
        // Circle animation
        switch (phase.text) {
            case "Breathe in":
                breathingCircle.style.transform = "scale(1)";
                break;
            case "Breathe out":
                breathingCircle.style.transform = "scale(0.6)";
                break;
            default:
                breathingCircle.style.transform = "scale(0.8)"; // hold
        }
    }
});

stopBreathingBtn.addEventListener("click", () => {
    clearInterval(breathingInterval);
    breathingContainer.classList.add("hidden");
});





// --------------------
// Power Nap Timer
// --------------------
const napButtons = document.querySelectorAll(".nap-btn");
const napTimerContainer = document.getElementById("napTimerContainer");
const napTimeDisplay = document.getElementById("napTimeDisplay");
const stopNapBtn = document.getElementById("stopNapBtn");

let napInterval;
let napTimeSeconds = 0;

napButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        clearInterval(napInterval);
        const minutes = parseInt(btn.dataset.minutes);
        napTimeSeconds = minutes * 60;
        napTimeDisplay.textContent = formatTime(napTimeSeconds);
        napTimerContainer.classList.remove("hidden");

        napInterval = setInterval(() => {
            napTimeSeconds--;
            napTimeDisplay.textContent = formatTime(napTimeSeconds);

            if (napTimeSeconds <= 0) {
                clearInterval(napInterval);
                napTimeDisplay.textContent = "Time's up!";
            }
        }, 1000);
    });
});

stopNapBtn.addEventListener("click", () => {
    clearInterval(napInterval);
    napTimerContainer.classList.add("hidden");
});

// --------------------
// Helper function: Format seconds into MM:SS
// --------------------
function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, "0");
    const s = (seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
}




// ===================== Music Section =====================
if (window.location.pathname.includes("recommendation.html")) {
    const youtubeInput = document.getElementById("youtubeLinkInput");
    const addYoutubeBtn = document.getElementById("addYoutubeBtn");
    const trackStatus = document.getElementById("trackStatus");

    const musicListContainer = document.getElementById("uploadedList");
    const playlistContainer = document.getElementById("playlistContainer");
    const playlistNameInput = document.getElementById("newPlaylistName");
    const createPlaylistBtn = document.getElementById("createPlaylistBtn");
    const playerContainer = document.getElementById("youtubePlayerContainer");

    let currentUser = localStorage.getItem("username") || "Guest";
    let currentPlaylistTracks = [];

    // -------------------- Notification System --------------------
    function showNotification(message, type = "success") {
        const box = document.getElementById("notificationBox");
        box.textContent = message;
        box.className = "notification show " + type;
        setTimeout(() => {
            box.className = "notification hidden";
        }, 2500);
    }

    // -------------------- Helpers --------------------
    function extractYouTubeID(url) {
        const regex = /(?:youtube\.com\/.*v=|youtu\.be\/)([^&\n?#]+)/;
        const match = url.match(regex);
        return match ? match[1] : null;
    }

    // -------------------- Load User Music --------------------
    async function loadMusic() {
        try {
            const res = await eel.get_user_music(currentUser)();
            musicListContainer.innerHTML = "";
            if (res.status === "success" && res.data.length > 0) {
                res.data.forEach(track => {
                    const div = document.createElement("div");
                    div.className = "uploaded-track";
                    div.innerHTML = `
                        <span>${track}</span>
                        <button class="playBtn">▶️ Play</button>
                        <button class="addToPlaylistBtn">➕</button>
                        <button class="deleteBtn">🗑️</button>
                    `;

                    // Play track
                    div.querySelector(".playBtn").onclick = () => {
                        const videoID = extractYouTubeID(track);
                        if (!videoID) return showNotification("Invalid YouTube URL", "error");

                        playerContainer.innerHTML = `
                            <iframe width="100%" height="250"
                                src="https://www.youtube.com/embed/${videoID}?autoplay=1"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen>
                            </iframe>
                        `;
                    };

                    // Add track to playlist
                    div.querySelector(".addToPlaylistBtn").onclick = () => {
                        if (!currentPlaylistTracks.includes(track)) {
                            currentPlaylistTracks.push(track);
                            showNotification(`Added to selection (${currentPlaylistTracks.length})`, "success");
                        } else {
                            showNotification("Track already added", "error");
                        }
                    };

                    // Delete track
                    div.querySelector(".deleteBtn").onclick = async () => {
                        const delRes = await eel.delete_youtube_track(currentUser, track)();
                        if (delRes.status === "success") {
                            div.remove();
                            currentPlaylistTracks = currentPlaylistTracks.filter(t => t !== track);
                            showNotification("Track removed!", "success");
                        } else {
                            showNotification(delRes.message, "error");
                        }
                    };

                    musicListContainer.appendChild(div);
                });
            } else {
                musicListContainer.innerHTML = "<p>No tracks uploaded.</p>";
            }
        } catch (err) {
            console.error("Error loading music:", err);
        }
    }

    // -------------------- Load Playlists --------------------
    async function loadPlaylists() {
        try {
            const res = await eel.get_playlists(currentUser)();
            playlistContainer.innerHTML = "";

            if (res.status === "success" && res.data.length > 0) {
                res.data.forEach(pl => {
                    const tracks = pl.tracks || [];
                    const div = document.createElement("div");
                    div.className = "playlist-card";
                    div.innerHTML = `
                        <span>${pl.playlist_name} (${tracks.length} tracks)</span>
                        <div>
                            <button class="viewBtn">View</button>
                            <button class="playPlaylistBtn">▶️ Play All</button>
                            <button class="deleteBtn">🗑️</button>
                        </div>
                    `;

                    // View playlist
                    div.querySelector(".viewBtn").onclick = () => {
                        if (tracks.length === 0)
                            showNotification(`Playlist "${pl.playlist_name}" is empty.`, "error");
                        else
                            showNotification(`Tracks:\n${tracks.join("\n")}`, "success");
                    };

                    // Play playlist
                    div.querySelector(".playPlaylistBtn").onclick = () => {
                        if (tracks.length === 0)
                            return showNotification("Playlist is empty.", "error");

                        const firstID = extractYouTubeID(tracks[0]);
                        if (!firstID)
                            return showNotification("Invalid URL", "error");

                        playerContainer.innerHTML = `
                            <iframe width="100%" height="250"
                                src="https://www.youtube.com/embed/${firstID}?autoplay=1"
                                allowfullscreen>
                            </iframe>
                        `;
                    };

                    // Delete playlist
                    div.querySelector(".deleteBtn").onclick = async () => {
                        const delRes = await eel.delete_playlist(currentUser, pl.playlist_name)();
                        if (delRes.status === "success") {
                            div.remove();
                            showNotification("Playlist deleted", "success");
                        } else {
                            showNotification(delRes.message, "error");
                        }
                    };

                    playlistContainer.appendChild(div);
                });
            } else {
                playlistContainer.innerHTML = "<p>No playlists found.</p>";
            }
        } catch (err) {
            console.error("Error loading playlists:", err);
        }
    }

    // -------------------- Add YouTube Track --------------------
    addYoutubeBtn.onclick = async () => {
        const url = youtubeInput.value.trim();
        if (!url) return showNotification("Enter a YouTube URL", "error");

        const ytRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
        if (!ytRegex.test(url)) return showNotification("Invalid YouTube URL", "error");

        try {
            const res = await eel.add_youtube_track(currentUser, url)();
            if (res.status === "success") {
                showNotification("Track added!", "success");
                youtubeInput.value = "";
                loadMusic();
            } else {
                showNotification(res.message, "error");
            }
        } catch (err) {
            console.error("Error adding track:", err);
        }
    };

    // -------------------- Create Playlist --------------------
    createPlaylistBtn.onclick = async () => {
        const name = playlistNameInput.value.trim();
        if (!name) return showNotification("Enter playlist name", "error");

        if (!currentPlaylistTracks.length)
            return showNotification("Add at least one track", "error");

        try {
            const res = await eel.create_playlist(currentUser, name, currentPlaylistTracks)();
            if (res.status === "success") {
                showNotification(`Playlist "${name}" created!`, "success");
                playlistNameInput.value = "";
                currentPlaylistTracks = [];
                loadMusic();
                loadPlaylists();
            } else {
                showNotification(res.message, "error");
            }
        } catch (err) {
            console.error("Error creating playlist:", err);
        }
    };

    // -------------------- Initialize --------------------
    loadMusic();
    loadPlaylists();
}




// ===================== Exercise Videos Section =====================
const exerciseVideos = [
    { title: "Jumping Jacks", url: "https://www.youtube.com/watch?v=-O7z3ilCu-s" },
    { title: "Shoulder Stretches", url: "https://www.youtube.com/watch?v=wnlcuZ0mJSU" },
    { title: "Hip Circles Exercise", url: "https://www.youtube.com/watch?v=Wh1Kg2iqBiw" }
];

const exerciseListContainer = document.getElementById("exerciseList");
const exercisePlayerContainer = document.getElementById("exercisePlayerContainer");

exerciseVideos.forEach(video => {
    const btn = document.createElement("button");
    btn.className = "exercise-btn";
    btn.textContent = video.title;

    btn.onclick = () => {
        // Only embed if direct video link
        const videoID = extractYouTubeID(video.url);
        if (videoID) {
            exercisePlayerContainer.innerHTML = `
                <iframe width="100%" height="250"
                    src="https://www.youtube.com/embed/${videoID}?autoplay=1"
                    title="YouTube video player" frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen>
                </iframe>
            `;
        } else {
            alert("Cannot directly play this video, please open it on YouTube: " + video.url);
        }
    };

    exerciseListContainer.appendChild(btn);
});



// ===================== Dance Exercise Videos Section =====================
const danceExerciseVideos = [
    { title: "5 min Dance Exercise", url: "https://www.youtube.com/watch?v=YKQIGAv7qb4&list=RDYKQIGAv7qb4&start_radio=1" },
    { title: "10 min Dance Exercise", url: "https://www.youtube.com/watch?v=m4zt9FGF3Ms&t=11s" },
    { title: "20 min Dance Exercise", url: "https://www.youtube.com/watch?v=IiaePoCD7zc" }
];

const danceListContainer = document.getElementById("danceExerciseList");
const dancePlayerContainer = document.getElementById("danceExercisePlayerContainer");

danceExerciseVideos.forEach(video => {
    const btn = document.createElement("button");
    btn.className = "exercise-btn";
    btn.textContent = video.title;

    btn.onclick = () => {
        const videoID = extractYouTubeID(video.url);
        if (videoID) {
            dancePlayerContainer.innerHTML = `
                <iframe width="100%" height="250"
                    src="https://www.youtube.com/embed/${videoID}?autoplay=1"
                    title="YouTube video player" frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen>
                </iframe>
            `;
        } else {
            alert("Cannot directly play this video, please open it on YouTube: " + video.url);
        }
    };

    danceListContainer.appendChild(btn);
});
