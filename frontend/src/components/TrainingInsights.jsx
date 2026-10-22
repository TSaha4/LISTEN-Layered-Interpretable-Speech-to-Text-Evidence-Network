import { memo, useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

function TrainingInsights({ apiBase }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetch(`${apiBase}/insights/training`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch training insights`);
        return res.json();
      })
      .then((json) => {
        if (active) setData(json);
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  if (loading) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P1</span> GAT Training Dynamics</div>
        <h2>Model Training & Architecture</h2>
        <div className="loading-state">
          <span className="spinner" /> Loading training checkpoint artifacts…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P1</span> GAT Training Dynamics</div>
        <h2>Model Training & Architecture</h2>
        <div className="error-box">
          <p><strong>Error loading training data:</strong> {error}</p>
        </div>
      </section>
    );
  }

  if (!data || !data.available) {
    return (
      <section className="panel insights-panel">
        <div className="panel-kicker"><span>P1</span> GAT Training Dynamics</div>
        <h2>Model Training & Architecture</h2>
        <div className="empty-box">
          <p>Training log artifact (<code>gat_train_log.json</code>) is not available on disk.</p>
        </div>
      </section>
    );
  }

  // Build chart points from epoch arrays
  const chartPoints = data.epochs.map((epoch, idx) => ({
    epoch: `Ep ${epoch}`,
    trainLoss: data.train_loss?.[idx] != null ? Number(data.train_loss[idx].toFixed(4)) : null,
    valLoss: data.val_loss?.[idx] != null ? Number(data.val_loss[idx].toFixed(4)) : null,
    valAccuracy: data.val_accuracy?.[idx] != null ? Number((data.val_accuracy[idx] * 100).toFixed(2)) : null,
    valF1: data.val_f1?.[idx] != null ? Number(data.val_f1[idx].toFixed(4)) : null,
  }));

  const arch = data.architecture || {};

  return (
    <section className="panel insights-panel">
      <div className="panel-kicker"><span>P1</span> GAT Training Dynamics</div>
      <div className="insights-header-row">
        <div>
          <h2>Training Dynamics & Architecture</h2>
          <p className="muted">
            Real loss curves, validation performance, and parameter counts loaded from <code>gat_train_log.json</code> and <code>gat_qmsum.pt</code>.
          </p>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <div className="stat-card">
          <span className="stat-label">Best Val F1</span>
          <span className="stat-value">
            {data.best_val_f1?.value != null ? Number(data.best_val_f1.value.toFixed(4)) : "not available"}
          </span>
          <span className="stat-sub">
            {data.best_val_f1?.epoch ? `Reached at Epoch ${data.best_val_f1.epoch}` : "—"}
          </span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Best Val Accuracy</span>
          <span className="stat-value">
            {data.best_val_accuracy?.value != null
              ? `${(data.best_val_accuracy.value * 100).toFixed(1)}%`
              : "not available"}
          </span>
          <span className="stat-sub">
            {data.best_val_accuracy?.epoch ? `Reached at Epoch ${data.best_val_accuracy.epoch}` : "—"}
          </span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Total Parameters</span>
          <span className="stat-value">
            {data.total_parameters != null ? data.total_parameters.toLocaleString() : "not available"}
          </span>
          <span className="stat-sub">EvidenceGAT architecture</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Trainable Parameters</span>
          <span className="stat-value">
            {data.trainable_parameters != null ? data.trainable_parameters.toLocaleString() : "not available"}
          </span>
          <span className="stat-sub">Full model gradients</span>
        </div>
      </div>

      {/* Architecture Strip */}
      <div className="architecture-strip">
        <div className="arch-item">
          <span className="arch-kicker">Input Dim</span>
          <span className="arch-val">{arch.input_dim ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">Hidden Dim</span>
          <span className="arch-val">{arch.hidden_dim ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">Output Dim</span>
          <span className="arch-val">{arch.output_dim ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">GAT Heads</span>
          <span className="arch-val">{arch.num_heads ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">GAT Layers</span>
          <span className="arch-val">{arch.num_layers ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">Dropout</span>
          <span className="arch-val">{arch.dropout ?? "not available"}</span>
        </div>
        <div className="arch-item">
          <span className="arch-kicker">Edge Types</span>
          <span className="arch-val">{arch.num_edge_types ?? "not available"}</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Loss Curves */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Training Loss Curves</h3>
            <span className="badge">Train vs Validation</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartPoints} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="epoch" tick={{ fill: "#665", fontSize: 11 }} />
                <YAxis tick={{ fill: "#665", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  name="Train Loss"
                  stroke="#e28c3e"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#e28c3e" }}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  name="Val Loss"
                  stroke="#2b6cb0"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#2b6cb0" }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption muted">
            Real epoch loss from <code>data/checkpoints/gat_train_log.json</code>.
          </p>
        </div>

        {/* Accuracy and F1 Chart */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Validation Accuracy & F1</h3>
            <span className="badge">GAT Classification</span>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartPoints} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6dfd5" />
                <XAxis dataKey="epoch" tick={{ fill: "#665", fontSize: 11 }} />
                <YAxis
                  yAxisId="left"
                  tick={{ fill: "#665", fontSize: 11 }}
                  unit="%"
                  domain={[40, 100]}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fill: "#665", fontSize: 11 }}
                  domain={[0, 0.4]}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#15120e", color: "#fff", borderRadius: 4, border: "none" }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="valAccuracy"
                  name="Val Accuracy (%)"
                  stroke="#2f855a"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#2f855a" }}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="valF1"
                  name="Val F1"
                  stroke="#c53030"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#c53030" }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption note-callout">
            <strong>Note:</strong> epoch-1 accuracy corresponds to F1 = 0 (all-negative predictions)
          </p>
        </div>
      </div>
    </section>
  );
}

export default memo(TrainingInsights);
