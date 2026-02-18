/**
 * ORION Dashboard — Frontend Controller
 *
 * Responsibilities
 * ----------------
 * - CSV upload handling for all three table types
 * - Anomaly detection: fetch, render Plotly chart
 * - Root cause analysis: fetch, render bar chart, populate table
 * - Impact simulation: fetch, render distribution chart, populate stats
 * - Recommendation: fetch and display Gemini-generated text
 * - Global error display and status indicator management
 */

"use strict";

/* -------------------------------------------------------------------------
   State
   ------------------------------------------------------------------------- */

const state = {
    loadedTables: new Set(),
    lastAnomalyResult: null,
    lastRootCauseResult: null,
    lastSimulationResult: null,
};

/* -------------------------------------------------------------------------
   Plotly layout defaults
   ------------------------------------------------------------------------- */

const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#e6edf3", size: 12 },
    margin: { t: 32, r: 24, b: 48, l: 60 },
    xaxis: {
        gridcolor: "#30363d",
        linecolor: "#30363d",
        zerolinecolor: "#30363d",
        tickfont: { color: "#8b949e" },
    },
    yaxis: {
        gridcolor: "#30363d",
        linecolor: "#30363d",
        zerolinecolor: "#30363d",
        tickfont: { color: "#8b949e" },
    },
    legend: {
        bgcolor: "#1c2333",
        bordercolor: "#30363d",
        borderwidth: 1,
    },
};

function buildLayout(overrides = {}) {
    return Object.assign({}, PLOTLY_LAYOUT_BASE, overrides);
}

/* -------------------------------------------------------------------------
   Status indicator
   ------------------------------------------------------------------------- */

const statusEl = document.getElementById("status-indicator");

function setStatus(mode, text) {
    statusEl.className = `status-${mode}`;
    statusEl.textContent = text;
}

/* -------------------------------------------------------------------------
   Global error toast
   ------------------------------------------------------------------------- */

const errorToast = document.getElementById("global-error");
let errorTimer = null;

function showError(message) {
    errorToast.textContent = message;
    errorToast.classList.remove("hidden");
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => errorToast.classList.add("hidden"), 6000);
}

/* -------------------------------------------------------------------------
   API helpers
   ------------------------------------------------------------------------- */

async function apiPost(endpoint, body) {
    const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const json = await response.json();
    if (!json.success) {
        throw new Error(json.error || "An unknown error occurred.");
    }
    return json.data;
}

async function apiUpload(endpoint, formData) {
    const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
    });
    const json = await response.json();
    if (!json.success) {
        throw new Error(json.error || "Upload failed.");
    }
    return json.data;
}

/* -------------------------------------------------------------------------
   CSV Upload
   ------------------------------------------------------------------------- */

document.querySelectorAll(".file-input").forEach(input => {
    input.addEventListener("change", async function () {
        const tableType = this.dataset.table;
        const file = this.files[0];
        if (!file) return;

        const statusEl = document.getElementById(`status-${tableType}`);
        statusEl.textContent = "Uploading...";
        statusEl.className = "upload-status";
        setStatus("busy", "Uploading data...");

        const formData = new FormData();
        formData.append("file", file);
        formData.append("table_type", tableType);

        try {
            const data = await apiUpload("/upload-data", formData);
            state.loadedTables.add(tableType);
            statusEl.textContent = `Loaded — ${data.rows} rows`;
            statusEl.className = "upload-status loaded";
            setStatus("idle", "System Ready");
        } catch (err) {
            statusEl.textContent = `Error: ${err.message}`;
            statusEl.className = "upload-status error";
            setStatus("error", "Upload Failed");
            showError(`Upload error (${tableType}): ${err.message}`);
        }
    });
});

/* -------------------------------------------------------------------------
   Anomaly Detection
   ------------------------------------------------------------------------- */

document.getElementById("btn-detect").addEventListener("click", async () => {
    const kpiName = document.getElementById("input-kpi-name").value.trim();
    if (!kpiName) {
        showError("Please enter a KPI name.");
        return;
    }

    setStatus("busy", "Running detection...");
    document.getElementById("btn-detect").disabled = true;

    try {
        const data = await apiPost("/detect-anomaly", { kpi_name: kpiName });
        state.lastAnomalyResult = data;
        renderAnomalyChart(data);
        const summaryEl = document.getElementById("anomaly-summary");
        const labelEl = document.getElementById("anomaly-count-label");
        labelEl.textContent = `${data.anomaly_count} anomalous point(s) detected out of ${data.total_points} total observations.`;
        summaryEl.classList.remove("hidden");
        setStatus("idle", "Detection Complete");
    } catch (err) {
        showError(`Detection error: ${err.message}`);
        setStatus("error", "Detection Failed");
    } finally {
        document.getElementById("btn-detect").disabled = false;
    }
});

function renderAnomalyChart(data) {
    const series = data.full_series || [];
    const dates = series.map(p => p.date);
    const values = series.map(p => p.value);
    const anomalyFlags = series.map(p => p.is_anomaly);

    const normalDates = dates.filter((_, i) => !anomalyFlags[i]);
    const normalValues = values.filter((_, i) => !anomalyFlags[i]);
    const anomalyDates = dates.filter((_, i) => anomalyFlags[i]);
    const anomalyValues = values.filter((_, i) => anomalyFlags[i]);

    const traceNormal = {
        x: normalDates,
        y: normalValues,
        mode: "lines+markers",
        name: "Normal",
        line: { color: "#3fb950", width: 1.5 },
        marker: { size: 4, color: "#3fb950" },
    };

    const traceAnomalies = {
        x: anomalyDates,
        y: anomalyValues,
        mode: "markers",
        name: "Anomaly",
        marker: { size: 10, color: "#f85149", symbol: "circle-open", line: { width: 2 } },
    };

    const layout = buildLayout({
        title: { text: `${data.kpi_name} — Anomaly Detection`, font: { size: 14 } },
        xaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.xaxis, { title: "Date" }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.yaxis, { title: "KPI Value" }),
        hovermode: "x unified",
    });

    Plotly.newPlot("chart-anomaly", [traceNormal, traceAnomalies], layout, { responsive: true, displayModeBar: false });
}

/* -------------------------------------------------------------------------
   Root Cause Analysis
   ------------------------------------------------------------------------- */

document.getElementById("btn-root-cause").addEventListener("click", async () => {
    const kpiName = document.getElementById("rc-kpi-name").value.trim();
    const anomalyDate = document.getElementById("rc-anomaly-date").value.trim() || null;

    if (!kpiName) {
        showError("Please enter a KPI name.");
        return;
    }

    setStatus("busy", "Analysing root causes...");
    document.getElementById("btn-root-cause").disabled = true;

    try {
        const data = await apiPost("/root-cause", { kpi_name: kpiName, anomaly_date: anomalyDate });
        state.lastRootCauseResult = data;
        renderRootCauseChart(data);
        renderRootCauseTable(data);
        setStatus("idle", "Analysis Complete");
    } catch (err) {
        showError(`Root cause error: ${err.message}`);
        setStatus("error", "Analysis Failed");
    } finally {
        document.getElementById("btn-root-cause").disabled = false;
    }
});

function renderRootCauseChart(data) {
    const causes = (data.ranked_causes || []).slice(0, 12);
    if (!causes.length) return;

    const labels = causes.map(c => c.segment);
    const values = causes.map(c => c.contribution);
    const colors = values.map(v => (v >= 0 ? "#3fb950" : "#f85149"));

    const trace = {
        type: "bar",
        orientation: "h",
        x: values,
        y: labels,
        marker: { color: colors },
        hovertemplate: "<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
    };

    const layout = buildLayout({
        title: { text: "Segment Contribution (SHAP-weighted)", font: { size: 14 } },
        xaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.xaxis, { title: "Weighted Contribution" }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.yaxis, { autorange: "reversed" }),
        height: 360,
    });

    Plotly.newPlot("chart-root-cause", [trace], layout, { responsive: true, displayModeBar: false });
}

function renderRootCauseTable(data) {
    const causes = data.ranked_causes || [];
    const tbody = document.getElementById("tbody-root-cause");
    tbody.innerHTML = "";

    causes.forEach((c, idx) => {
        const row = document.createElement("tr");
        const direction = c.contribution >= 0
            ? '<span class="badge-positive">Positive</span>'
            : '<span class="badge-negative">Negative</span>';
        row.innerHTML = `
            <td>${idx + 1}</td>
            <td>${escapeHtml(c.segment)}</td>
            <td>${c.contribution.toFixed(5)}</td>
            <td>${direction}</td>
        `;
        tbody.appendChild(row);
    });

    document.getElementById("table-root-cause").classList.remove("hidden");
}

/* -------------------------------------------------------------------------
   Impact Simulation
   ------------------------------------------------------------------------- */

document.getElementById("btn-simulate").addEventListener("click", async () => {
    const kpiName = document.getElementById("sim-kpi-name").value.trim();
    const segmentName = document.getElementById("sim-segment-name").value.trim();
    const changePct = parseFloat(document.getElementById("sim-change-pct").value);

    if (!kpiName || !segmentName) {
        showError("KPI name and segment name are required.");
        return;
    }

    if (isNaN(changePct)) {
        showError("Change percentage must be a valid number.");
        return;
    }

    setStatus("busy", "Running simulation...");
    document.getElementById("btn-simulate").disabled = true;

    try {
        const data = await apiPost("/simulate-impact", {
            kpi_name: kpiName,
            segment_name: segmentName,
            change_pct: changePct,
        });
        state.lastSimulationResult = data;
        renderSimulationStats(data);
        renderSimulationChart(data);
        document.getElementById("sim-results").classList.remove("hidden");
        setStatus("idle", "Simulation Complete");
    } catch (err) {
        showError(`Simulation error: ${err.message}`);
        setStatus("error", "Simulation Failed");
    } finally {
        document.getElementById("btn-simulate").disabled = false;
    }
});

function renderSimulationStats(data) {
    document.getElementById("sim-current-kpi").textContent = fmt(data.current_kpi);
    document.getElementById("sim-regression").textContent = fmt(data.regression_estimate);
    document.getElementById("sim-p50").textContent = fmt(data.p50);
    document.getElementById("sim-range").textContent = `${fmt(data.p10)} — ${fmt(data.p90)}`;
}

function renderSimulationChart(data) {
    const mean = data.simulated_mean;
    const std = data.simulated_std;
    const n = 200;
    const xs = [];
    const ys = [];

    for (let i = 0; i < n; i++) {
        const x = mean - 4 * std + (i / (n - 1)) * 8 * std;
        xs.push(x);
        ys.push(gaussian(x, mean, std));
    }

    const traceDistribution = {
        x: xs,
        y: ys,
        mode: "lines",
        name: "Projected Distribution",
        fill: "tozeroy",
        fillcolor: "rgba(47, 129, 247, 0.15)",
        line: { color: "#2f81f7", width: 2 },
    };

    const traceP50 = verticalLine(data.p50, "#3fb950", "P50");
    const traceP10 = verticalLine(data.p10, "#d29922", "P10");
    const traceP90 = verticalLine(data.p90, "#d29922", "P90");
    const traceCurrent = verticalLine(data.current_kpi, "#8b949e", "Current");

    const layout = buildLayout({
        title: { text: `${data.segment_name} +${data.change_pct}% — KPI Impact Distribution`, font: { size: 14 } },
        xaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.xaxis, { title: "Projected KPI Value" }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.yaxis, { title: "Probability Density" }),
        hovermode: "x",
    });

    Plotly.newPlot("chart-simulation", [traceDistribution, traceP10, traceP50, traceP90, traceCurrent], layout, { responsive: true, displayModeBar: false });
}

function gaussian(x, mean, std) {
    if (std === 0) return 0;
    return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mean) / std) ** 2);
}

function verticalLine(x, color, name) {
    return {
        x: [x, x],
        y: [0, 1],
        yaxis: "y",
        mode: "lines",
        name: name,
        line: { color: color, width: 1.5, dash: "dash" },
        hovertemplate: `${name}: ${fmt(x)}<extra></extra>`,
    };
}

/* -------------------------------------------------------------------------
   Recommendations
   ------------------------------------------------------------------------- */

document.getElementById("btn-recommend").addEventListener("click", async () => {
    const context = buildRecommendationContext();
    if (!context || Object.keys(context).length === 0) {
        showError("Run at least one analysis step before generating a recommendation.");
        return;
    }

    setStatus("busy", "Generating recommendation...");
    document.getElementById("btn-recommend").disabled = true;

    try {
        const data = await apiPost("/recommendation", { context });
        const outputBox = document.getElementById("recommendation-output");
        document.getElementById("recommendation-text").textContent = data.recommendation;
        outputBox.classList.remove("hidden");
        setStatus("idle", "Recommendation Ready");
    } catch (err) {
        showError(`Recommendation error: ${err.message}`);
        setStatus("error", "Recommendation Failed");
    } finally {
        document.getElementById("btn-recommend").disabled = false;
    }
});

function buildRecommendationContext() {
    const ctx = {};

    if (state.lastAnomalyResult) {
        ctx.anomaly_detection = {
            kpi_name: state.lastAnomalyResult.kpi_name,
            anomaly_count: state.lastAnomalyResult.anomaly_count,
            total_points: state.lastAnomalyResult.total_points,
        };
    }

    if (state.lastRootCauseResult) {
        ctx.root_cause = {
            top_causes: (state.lastRootCauseResult.ranked_causes || []).slice(0, 5),
        };
    }

    if (state.lastSimulationResult) {
        ctx.simulation = {
            segment_name: state.lastSimulationResult.segment_name,
            change_pct: state.lastSimulationResult.change_pct,
            current_kpi: state.lastSimulationResult.current_kpi,
            regression_estimate: state.lastSimulationResult.regression_estimate,
            p50: state.lastSimulationResult.p50,
        };
    }

    return ctx;
}

/* -------------------------------------------------------------------------
   Helpers
   ------------------------------------------------------------------------- */

function fmt(value) {
    if (value === null || value === undefined) return "—";
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}
