"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  schema: null,
  kernel: null,
  playing: false,
  busy: false,
  probe: { px: 128, py: 96 },
  viewVersion: null,
  versions: [],
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-json */ }
  if (!res.ok || (data && data.ok === false)) {
    const err = new Error((data && data.error) || `HTTP ${res.status}`);
    err.payload = data;
    throw err;
  }
  return data;
}

function post(cmd, extra = {}) {
  return api("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd, ...extra }),
  });
}

function showError(msg) {
  const bar = $("errorBar");
  if (!msg) { bar.classList.add("hidden"); bar.textContent = ""; return; }
  bar.classList.remove("hidden");
  bar.textContent = "⚠ " + msg;
  $("statusDot").className = "dot err";
  $("statusText").textContent = "内核错误（已反馈到引导面板）";
}

function reportError(e) {
  showError(e && e.message ? e.message : String(e));
  // Command failures carry the latest kernel state so the failure is also
  // visible immediately in the "guiding kernel feedback" panel.
  const st = e && e.payload && e.payload.state;
  if (st) { try { applyState(st); } catch (_) { /* noop */ } }
}

/* ------------------------------------------------------------ schema UI */
function buildSchemaUI(schema) {
  state.schema = schema;
  // scenes
  const sb = $("sceneButtons");
  sb.innerHTML = "";
  schema.scenes.forEach((sc) => {
    const b = document.createElement("button");
    b.textContent = sc.label;
    b.dataset.kind = sc.value;
    b.onclick = async () => {
      try {
        showError(null);
        const r = await post("set_scene", { kind: sc.value });
        applyState(r.result);
        syncSceneActive();
        await refreshImage();
      } catch (e) { reportError(e); }
    };
    sb.appendChild(b);
  });
  // strategies
  const sel = $("strategySelect");
  sel.innerHTML = "";
  schema.strategies.forEach((st) => {
    const o = document.createElement("option");
    o.value = st.value; o.textContent = st.label;
    sel.appendChild(o);
  });
  sel.onchange = async () => {
    try {
      const r = await post("set", { params: { strategy: sel.value } });
      applyState(r.result);
    } catch (e) { reportError(e); }
  };
  // sliders
  const wrap = $("sliders");
  wrap.innerHTML = "";
  schema.controls.forEach((c) => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.dataset.key = c.key;
    if (c.type === "toggle") {
      row.classList.add("toggle-row");
      row.innerHTML = `
        <label class="toggle-lab">
          <input type="checkbox" id="chk_${c.key}" ${c.default ? "checked" : ""}/>
          <span>${c.label}</span>
        </label>
        ${c.hint ? `<p class="hint">${c.hint}</p>` : ""}`;
      wrap.appendChild(row);
      const chk = row.querySelector("input");
      chk.onchange = async () => {
        try {
          showError(null);
          const r = await post("set", { params: { [c.key]: chk.checked } });
          applyState(r.result);
        } catch (e) { reportError(e); }
      };
      return;
    }
    row.innerHTML = `
      <div class="lab"><span>${c.label}</span><span class="val" id="val_${c.key}"></span></div>
      <input type="range" id="rng_${c.key}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.default}"/>
      ${c.hint ? `<p class="hint">${c.hint}</p>` : ""}`;
    wrap.appendChild(row);
    const rng = row.querySelector("input");
    const val = row.querySelector(".val");
    const fmt = (v) => c.step < 0.1 ? Number(v).toFixed(2) : (c.step < 1 ? Number(v).toFixed(1) : v);
    val.textContent = fmt(c.default);
    let t = null;
    rng.oninput = () => {
      val.textContent = fmt(rng.value);
      clearTimeout(t);
      t = setTimeout(async () => {
        try {
          showError(null);
          const r = await post("set", { params: { [c.key]: Number(rng.value) } });
          applyState(r.result);
        } catch (e) { reportError(e); }
      }, 60);
    };
  });
}

function syncControlValues(k) {
  document.querySelectorAll(".slider-row").forEach((row) => {
    const key = row.dataset.key;
    const input = row.querySelector("input");
    if (input && input.type === "checkbox") {
      if (k[key] !== undefined && document.activeElement !== input) {
        input.checked = !!k[key];
      }
    } else {
      const rng = input;
      const val = row.querySelector(".val");
      if (rng && k[key] !== undefined && document.activeElement !== rng) {
        rng.value = k[key];
        if (val) val.textContent = Number(k[key]).toFixed(rng.step < 0.1 ? 2 : 1);
      }
    }
    const schemaCtl = state.schema.controls.find((c) => c.key === key);
    if (schemaCtl && schemaCtl.visible_when) {
      const visible = Object.entries(schemaCtl.visible_when)
        .every(([kk, vv]) => k.strategy === vv);
      row.style.display = visible ? "" : "none";
    }
  });
  $("strategySelect").value = k.strategy;
  const st = state.schema.strategies.find((s) => s.value === k.strategy);
  $("strategyDesc").textContent = st ? st.desc : "";
  document.querySelectorAll("#sceneButtons button").forEach((b) =>
    b.classList.toggle("active", b.dataset.kind === k.kind));
}

function syncSceneActive() { /* updated via applyState */ }

/* -------------------------------------------------- acceptance / regression */
function buildRegressionUI(schema) {
  const wrap = $("regressionCases");
  wrap.innerHTML = "";
  (schema.regression_cases || []).forEach((c) => {
    const b = document.createElement("button");
    b.className = "btn tiny";
    b.textContent = c.label;
    b.title = c.desc || "";
    b.onclick = async () => {
      setPlaying(false);
      try {
        showError(null);
        const r = await post("load_regression_case", { case: c.case });
        applyState(r.result);
        await refreshImage();
        await refreshDistribution();
      } catch (e) { reportError(e); }
    };
    wrap.appendChild(b);
  });
  $("acceptanceBtn").onclick = async () => {
    const btn = $("acceptanceBtn");
    btn.disabled = true;
    btn.textContent = "⏳ 验收运行中（真实采样，约十秒）…";
    try {
      showError(null);
      const r = await post("run_acceptance");
      applyState(r.result);
      await loadAcceptance();
      document.querySelector('.tab[data-tab="acceptance"]').click();
    } catch (e) { reportError(e); }
    finally {
      btn.disabled = false;
      btn.textContent = "▶ 运行统计正确性验收";
    }
  };
}

function ciText(ci, ref) {
  if (!ci) return "";
  const fmt = (v) => Number(v).toFixed(4);
  return `均值 ${fmt(ci.mean)}，95% CI [${fmt(ci.lo)}, ${fmt(ci.hi)}]` +
    (ref !== undefined ? `，独立参照 ${fmt(ref)}` : "") +
    (ci.n ? `，n=${ci.n}` : "");
}

async function loadAcceptance() {
  const tab = $("tab-acceptance");
  let data;
  try { data = await api("/api/acceptance"); }
  catch (e) {
    tab.innerHTML = `<p class="notice">验收报告不可用：${escapeHtml(e.message)}</p>`;
    return;
  }
  if (!data.available) {
    tab.innerHTML = `<p class="notice">尚未生成验收报告。点击左侧
      <b>运行统计正确性验收</b>（真实重算，非固定数据），或执行
      <code>python3 -m dirguide.acceptance</code>。</p>`;
    return;
  }
  const head = `<p class="desc">生成于 ${escapeHtml(data.generated_at)} · 用时
    ${data.elapsed_s}s · all_ok =
    <b style="color:${data.all_ok ? "#46d19a" : "#ff6b6b"}">${data.all_ok}</b>。
    均值置信区间来自独立试验/区间，参照值由解析小场景闭式推导；每项附执行证据。</p>`;
  const rows = (data.checks || []).map((c) => {
    const ok = !!c.ok;
    const ci = c.grand_ci || c.mean_ci ||
      (c.mean_image_ci) || (c.difference_ci) || null;
    const ref = c.reference_luminance ??
      (c.mean_image_ci && c.mean_image_ci.reference) ??
      (c.grand_ci && c.reference_luminance) ??
      (c.difference_ci ? 0.0 : undefined);
    let extra = "";
    if (c.verdict) extra += ` verdict=${escapeHtml(c.verdict)}`;
    if (c.missing_energy_fraction !== undefined)
      extra += ` 遗漏能量 ${(100 * c.missing_energy_fraction).toFixed(1)}%`;
    if (c.variance_ratio !== undefined)
      extra += ` 方差比 ${Number(c.variance_ratio).toFixed(2)}`;
    if (c.chi2 !== undefined)
      extra += ` chi2=${Number(c.chi2).toFixed(1)} df=${c.df}`;
    if (c.first_anomalous_component)
      extra += ` 首个异常光源分量=${escapeHtml(c.first_anomalous_component.light)}` +
        `@区间${c.first_anomalous_component.interval}`;
    const ev = ci && ci.evidence
      ? `<span class="desc">证据(试验均值): ${ci.evidence.map((v) => Number(v).toFixed(4)).join(", ")}</span><br>`
      : "";
    return `<div class="check-item ${ok ? "ok" : "bad"}">` +
      `${ok ? "✅" : "❌"} <b>${escapeHtml(c.name)}</b>${extra}<br>` +
      (ci ? `<span class="desc">${ciText(ci, ref)}</span><br>` : "") +
      ev + `</div>`;
  }).join("");
  tab.innerHTML = head + rows;
}


/* ------------------------------------------------------------- rendering */
async function refreshImage() {
  const img = $("renderImg");
  img.src = `/api/image.png?t=${Date.now()}_${state.kernel ? state.kernel.spp : 0}`;
}

async function oneStep() {
  if (state.busy) return;
  state.busy = true;
  try {
    const r = await post("step", { spp: 1 });
    applyState(r.result);
    await refreshImage();
    if (state.probe) await refreshDistribution();
  } catch (e) {
    reportError(e);
    setPlaying(false);
  } finally {
    state.busy = false;
  }
}

async function playLoop() {
  if (!state.playing) return;
  await oneStep();
  if (state.playing) setTimeout(playLoop, 10);
}

function setPlaying(on) {
  state.playing = on;
  const b = $("playBtn");
  b.textContent = on ? "⏸ 暂停渲染" : "▶ 连续渲染";
  b.classList.toggle("playing", on);
}

/* ---------------------------------------------------------------- state */
function applyState(k) {
  state.kernel = k;
  syncControlValues(k);
  $("statusDot").className = "dot ok";
  $("statusText").textContent = `内核运行中 · ${k.kind} · v${k.guide_version}`;
  renderHud(k);
  renderStats(k);
  drawCharts(k);
  renderFeedback(k);
}

function renderHud(k) {
  $("hud").textContent =
    `${k.width}×${k.height}  spp=${k.spp}\n` +
    `${k.strategy_label}\n` +
    `光源 ${k.placement || "-"}  表版本 v${k.guide_version} (估计用 v${k.used_version_last ?? 0})\n` +
    `λ=${k.guide_mix}  η=${k.eta}` + (k.has_reference ? "\n参考图已载入" : "\n无参考图（运行 run_experiments.py）");
}

function chip(label, value, cls = "") {
  return `<span class="chip ${cls}">${label} <b>${value}</b></span>`;
}

function renderStats(k) {
  const last = k.last || {};
  const chips = [
    chip("累计 spp", k.spp),
    chip("用时", k.elapsed + " s"),
    chip("每 spp", (last.step_ms ?? "-") + " ms"),
    chip("追踪光线", (k.total_trace_rays / 1e6).toFixed(2) + "M"),
    chip("训练沉积", (k.n_deposits / 1000).toFixed(1) + "k"),
    chip("光代", k.light_generation, k.light_generation > 0 ? "warn" : ""),
  ];
  if (last.rmse_display !== undefined)
    chips.push(chip("RMSE", Number(last.rmse_display).toFixed(4), "good"));
  if (last.rel_mse !== undefined)
    chips.push(chip("relMSE", Number(last.rel_mse).toExponential(2), "good"));
  if (last.mean_luma !== undefined)
    chips.push(chip("平均亮度", Number(last.mean_luma).toFixed(3)));
  $("statsBar").innerHTML = chips.join("");
  $("rmseNote").textContent = k.has_reference
    ? ""
    : "尚未找到匹配的高采样参考图；执行 python3 run_experiments.py 后这里显示 RMSE 收敛。";
}

function renderFeedback(k) {
  const box = $("feedback");
  const items = (k.feedback || []).slice().reverse();
  box.innerHTML = items.map((f) =>
    `<div class="fb ${f.kind}">[${f.kind}] ${escapeHtml(f.message)}</div>`).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* --------------------------------------------------------------- charts */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || cv.width;
  const cssH = cv.clientHeight || 180;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

function drawLineChart(canvasId, points, yKey, color, label) {
  const cv = $(canvasId);
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const padL = 46, padR = 12, padT = 14, padB = 26;
  const pw = w - padL - padR, ph = h - padT - padB;
  ctx.strokeStyle = "#313a49"; ctx.fillStyle = "#9aa7b8";
  ctx.font = "10px ui-monospace, monospace"; ctx.lineWidth = 1;
  // grid + y range
  let ys = points.map((p) => p[yKey]).filter((v) => v !== undefined && isFinite(v));
  if (!ys.length) {
    ctx.fillStyle = "#9aa7b8"; ctx.fillText("等待数据 / 参考图", padL, padT + 20);
    return;
  }
  let ymin = 0, ymax = Math.max(...ys) * 1.08;
  if (yKey === "mean_luma") { ymin = Math.min(...ys) * 0.96; ymax = Math.max(...ys) * 1.04; }
  const xmax = Math.max(...points.map((p) => p.spp));
  const X = (s) => padL + (xmax <= 1 ? 0 : (s / xmax) * pw);
  const Y = (v) => padT + ph - ((v - ymin) / ((ymax - ymin) || 1)) * ph;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke();
    const val = ymax - ((ymax - ymin) * i) / 4;
    ctx.fillText(val.toFixed(yKey === "mean_luma" ? 2 : 3), 6, y + 3);
  }
  ctx.fillText("spp →", padL + pw - 30, h - 8);
  // light-move marker
  let lastGen = 0;
  points.forEach((p) => {
    if (p.light_generation > lastGen) {
      lastGen = p.light_generation;
      if (p.spp > 1) {
        ctx.strokeStyle = "#f0b35088";
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(X(p.spp), padT); ctx.lineTo(X(p.spp), padT + ph); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  });
  // line
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  points.forEach((p) => {
    const v = p[yKey];
    if (v === undefined || !isFinite(v)) return;
    const x = X(p.spp), y = Y(v);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(label, padL + 6, padT + 2);
}

function drawCharts(k) {
  const hist = k.history || [];
  drawLineChart("rmseChart", hist, "rmse_display", "#5aa9ff", "RMSE display");
  drawLineChart("meanChart", hist, "mean_luma", "#46d19a", "mean luminance");
}

/* -------------------------------------------------------- distribution */
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [12, 16, 24], [30, 58, 138], [34, 160, 200],
    [70, 210, 160], [240, 220, 90], [235, 90, 70],
  ];
  const x = t * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * f)).join(",")})`;
}

async function refreshDistribution() {
  if (!state.probe) return;
  const { px, py } = state.probe;
  let q = `/api/distribution?px=${px}&py=${py}`;
  if (state.viewVersion) q += `&version=${encodeURIComponent(state.viewVersion)}`;
  try {
    const d = await api(q);
    state.lastDist = d.distribution;
    drawDistribution(state.lastDist);
  } catch (e) { reportError(e); }
}

function drawDistribution(d) {
  const cv = $("distCanvas");
  const dpr = window.devicePixelRatio || 1;
  const Wcss = cv.clientWidth || 380, Hcss = 210;
  cv.width = Wcss * dpr; cv.height = Hcss * dpr;
  cv.style.height = Hcss + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0e13"; ctx.fillRect(0, 0, Wcss, Hcss);
  const padL = 34, padR = 8, padT = 12, padB = 20;
  const pw = Wcss - padL - padR, ph = Hcss - padT - padB;

  if (!d || !d.hit) {
    ctx.fillStyle = "#9aa7b8"; ctx.font = "13px sans-serif";
    ctx.fillText("该像素未命中几何，请点击室内表面。", padL, 60);
    $("distMetrics").innerHTML = "";
    $("probeInfo").textContent = "未命中。";
    return;
  }
  const T = d.theta_bins, P = d.phi_bins;
  const mass = d.g_mass;
  const domega = d.bin_domega;
  const dens = mass.map((m, i) => m / domega[i]);
  const logs = dens.map((v) => Math.log10(Math.max(v, 1e-6)));
  const lo = Math.log10(d.floor_pdf || 1e-3);
  const hi = Math.max(...logs);
  const cw = pw / P, ch = ph / T;
  for (let t = 0; t < T; t++) {
    for (let p = 0; p < P; p++) {
      const b = t * P + p;
      const norm = (logs[b] - lo) / Math.max(hi - lo, 1e-6);
      ctx.fillStyle = heatColor(norm);
      ctx.fillRect(padL + p * cw, padT + t * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
    }
  }
  // axes labels
  ctx.fillStyle = "#cdd8e6"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("+Y", 8, padT + 8);
  ctx.fillText("−Y", 8, padT + ph - 4);
  ctx.fillText("0", padL, Hcss - 6);
  ctx.fillText("π", padL + pw / 2 - 4, Hcss - 6);
  ctx.fillText("2π", padL + pw - 18, Hcss - 6);

  const Xphi = (phi) => padL + (phi / (2 * Math.PI)) * pw;
  const Ytheta = (th) => padT + (th / Math.PI) * ph;

  // horizon curve for the hit normal: sweep the great circle n·ω = 0 via an
  // orthonormal basis, which stays correct for horizontal normals (where a
  // single-valued theta(phi) solution degenerates).
  const n = d.normal;
  if (n) {
    const nl = Math.hypot(n[0], n[1], n[2]) || 1;
    const nn = [n[0] / nl, n[1] / nl, n[2] / nl];
    const a = Math.abs(nn[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const cross = (u, v) => [
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0],
    ];
    const norm = (u) => {
      const l = Math.hypot(u[0], u[1], u[2]) || 1;
      return [u[0] / l, u[1] / l, u[2] / l];
    };
    const u = norm(cross(nn, a));
    const v = cross(nn, u);
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    let pen = false;
    let prevPhi = null;
    for (let i = 0; i <= 240; i++) {
      const psi = (i / 240) * 2 * Math.PI;
      const w = [
        Math.cos(psi) * u[0] + Math.sin(psi) * v[0],
        Math.cos(psi) * u[1] + Math.sin(psi) * v[1],
        Math.cos(psi) * u[2] + Math.sin(psi) * v[2],
      ];
      const theta = Math.acos(Math.max(-1, Math.min(1, w[1])));
      let phi = Math.atan2(w[2], w[0]);
      if (phi < 0) phi += 2 * Math.PI;
      const x = Xphi(phi), y = Ytheta(theta);
      // Lift the pen across phi wrap / pole singularities so no spurious
      // horizontal span connects the branches.
      if (!pen || (prevPhi !== null && Math.abs(phi - prevPhi) > Math.PI / 2)) {
        ctx.moveTo(x, y);
        pen = true;
      } else {
        ctx.lineTo(x, y);
      }
      prevPhi = phi;
    }
    ctx.stroke();
    // normal pole
    const thN = Math.acos(Math.max(-1, Math.min(1, n[1])));
    let phN = Math.atan2(n[2], n[0]); if (phN < 0) phN += 2 * Math.PI;
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000";
    ctx.beginPath(); ctx.arc(Xphi(phN), Ytheta(thN), 4, 0, 7); ctx.fill(); ctx.stroke();
  }
  // peak marker
  let peak = 0;
  d.raw_weights.forEach((w, i) => { if (w > d.raw_weights[peak]) peak = i; });
  if (d.w_sum > 0) {
    const t = Math.floor(peak / P), p = peak % P;
    ctx.strokeStyle = "#00ffd0"; ctx.lineWidth = 2;
    ctx.strokeRect(padL + p * cw + 1, padT + t * ch + 1, cw - 2, ch - 2);
  }

  $("probeInfo").innerHTML =
    `像素 [${d.pixel[0]}, ${d.pixel[1]}] → 网格 <b>(${d.ijk.join(",")})</b>` +
    (d.version_file ? ` · 版本 <b>${escapeHtml(d.version_file)}</b>` : " · 实时表");

  const m = [];
  m.push(metric("观察权重和", d.w_sum.toFixed(1)));
  m.push(metric("归一化熵", d.entropy_norm.toFixed(3), d.entropy_norm > 0.8 ? "ok" : ""));
  m.push(metric("相对均匀 KL", d.kl_uniform.toFixed(3)));
  m.push(metric("最小 g pdf", d.min_pdf.toExponential(2)));
  m.push(metric("均匀底保证", d.floor_pdf.toExponential(2), "ok"));
  if (d.min_mixture_pdf_hemi_centers !== undefined)
    m.push(metric("半球最小混合 p", d.min_mixture_pdf_hemi_centers.toExponential(2), "ok"));
  if (d.strategy_lam !== undefined)
    m.push(metric("当前 λ", Number(d.strategy_lam).toFixed(2)));
  $("distMetrics").innerHTML = m.join("");

  // marker on image
  const mk = $("probeMarker");
  mk.classList.remove("hidden");
  mk.style.left = (100 * d.pixel[0] / state.kernel.width) + "%";
  mk.style.top = (100 * d.pixel[1] / state.kernel.height) + "%";
}

function metric(k, v, cls = "") {
  return `<div class="metric ${cls}"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

/* -------------------------------------------------------------- versions */
async function refreshVersions(selectName) {
  try {
    const r = await api("/api/versions");
    state.versions = r.versions;
    const sel = $("versionSelect");
    sel.innerHTML = `<option value="">— 选择已保存版本 —</option>` +
      r.versions.map((v) => `<option value="${escapeHtml(v.filename)}">` +
        `${escapeHtml(v.filename)}${v.label ? " · " + escapeHtml(v.label) : ""}</option>`).join("");
  } catch (e) { /* non-fatal */ }
}

/* --------------------------------------------------------------- results */
const KEY_LABELS = {
  strategy: "策略", policy: "移灯策略", label: "名称", kind: "场景",
  rmse_display: "RMSE(显示域)", rel_mse: "relMSE(HDR)", mean_luma: "平均亮度",
  mean_bias: "均值偏差", rmse_final: "最终RMSE", rmse_8: "RMSE@8",
  rmse_16: "RMSE@16", rmse_32: "RMSE@32", rmse_48: "RMSE@48",
  rmse_64: "RMSE@64", speedup_rmse: "RMSE降幅",
  rmse_hard_roi: "困难间接区RMSE", rmse_std: "RMSE种子标准差",
  train_spp: "训练spp", test_spp: "移灯后spp",
};

function renderTable(rows) {
  if (!rows || !rows.length) return "<p class='notice'>无数据</p>";
  const keys = Object.keys(rows[0]).filter((k) => k !== "src");
  const head = keys.map((k) => `<th>${KEY_LABELS[k] || k}</th>`).join("");
  const body = rows.map((r) => "<tr>" + keys.map((k) => {
    let v = r[k];
    if (typeof v === "number") {
      if (k.includes("mse") || k.includes("rmse") || k.includes("bias"))
        v = Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(4);
      else v = v.toFixed(3);
    }
    return `<td>${v ?? ""}</td>`;
  }).join("") + "</tr>").join("");
  return `<table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function figures(items) {
  if (!items) return "";
  return `<div class="figrow">${items.map((f) => `
    <figure><img src="${f.src}" alt="${escapeHtml(f.caption || "")}" loading="lazy"/>
    <figcaption>${escapeHtml(f.caption || "")}</figcaption></figure>`).join("")}</div>`;
}

async function loadResults() {
  let data;
  try { data = await api("/api/results"); }
  catch (e) {
    ["fixed", "move", "coverage", "files"].forEach((t) =>
      $("tab-" + t).innerHTML = `<p class="notice">结果服务不可用：${escapeHtml(e.message)}</p>`);
    return;
  }
  if (!data.available) {
    const msg = `<p class="notice">尚未生成实验结果。请在 /workspace 运行
      <b>python3 run_experiments.py</b>（首次会渲染高采样参考图，约数分钟），完成后刷新本页。</p>`;
    $("tab-fixed").innerHTML = msg;
    $("tab-move").innerHTML = msg;
    $("tab-coverage").innerHTML = msg;
    $("tab-files").innerHTML = msg;
    return;
  }
  // fixed
  $("tab-fixed").innerHTML = (data.fixed || []).map((b) => `
    <h3>${escapeHtml(b.label)} · 相同 ${b.budget_spp} spp 预算（${b.seeds} 个随机种子均值）</h3>
    ${renderTable(b.rows)}
    ${figures([{ src: b.panel, caption: "等预算渲染与误差热图" },
               { src: b.chart, caption: "RMSE / relMSE 收敛曲线" },
               b.dist_fig ? { src: b.dist_fig, caption: "选定位置学到的方向分布" } : null].filter(Boolean))}
    ${b.ablation ? `
      <h4>重要性采样隔离对照：关闭 NEE（${b.ablation.spp} spp，λ=${b.ablation.lam}，单种子）</h4>
      <p class="desc">关闭下一事件估计后，方向采样成为命中光源的唯一途径，可纯净检验“只改变重要性采样”的方差收益；估计仍使用同一个真实混合 pdf，故保持无偏。显示域 RMSE 相对降幅
        <b style="color:${b.ablation.rmse_reduction_pct >= 0 ? "var(--accent2)" : "#ff8a8a"}">${b.ablation.rmse_reduction_pct}%</b>。</p>
      ${renderTable(b.ablation.rows)}
      ${figures([{ src: b.ablation.chart, caption: "NEE-off：RMSE / relMSE 收敛对比" }])}
    ` : ""}
  `).join("") || "<p class='notice'>无</p>";
  // move
  $("tab-move").innerHTML = (data.move || []).map((b) => `
    <h3>${escapeHtml(b.label)} · 先训练 ${b.train_spp} spp，移灯后 ${b.test_spp} spp</h3>
    <p class="desc">三种策略在移灯时拥有完全相同的方向表；移灯后图像估计清零重启，方向表分别保留 / 衰减 / 重建。</p>
    ${renderTable(b.rows)}
    ${figures([{ src: b.panel, caption: "移灯后等预算结果与误差" },
               { src: b.chart, caption: "移灯后收敛对比" }])}
  `).join("") || "<p class='notice'>无</p>";
  // coverage
  const cov = data.coverage;
  if (cov) {
    $("tab-coverage").innerHTML =
      `<p class="desc">all_ok = <b style="color:${cov.all_ok ? "#46d19a" : "#ff6b6b"}">${cov.all_ok}</b>。
       这些数值验证真实混合概率、归一化、均匀底支撑与低概率方向覆盖。</p>` +
      cov.checks.map((c) => `<div class="check-item ${c.ok ? "ok" : "bad"}">` +
        `${c.ok ? "✅" : "❌"} ${escapeHtml(c.name)}<br><span class="desc">${escapeHtml(JSON.stringify(c))}</span></div>`).join("");
  }
  // files / gallery
  const gal = (data.gallery || []).map((src) =>
    `<div><a href="${src}" target="_blank"><img src="${src}" loading="lazy"/></a>
     <a href="${src}" target="_blank">${escapeHtml(src.split("/").pop())}</a></div>`).join("");
  $("tab-files").innerHTML =
    `<h3>全部图像产物</h3><div class="gallery">${gal}</div>
     <h3>分布版本 JSON</h3><div id="fileVersions" class="filelist">通过左侧“保存当前分布版本”生成，位于 outputs/versions/。</div>`;
  refreshVersions();
}

/* --------------------------------------------------------------- events */
function bindStatic() {
  $("playBtn").onclick = () => { setPlaying(!state.playing); if (state.playing) playLoop(); };
  $("stepBtn").onclick = () => oneStep();
  $("resetBtn").onclick = async () => {
    setPlaying(false);
    try { const r = await post("reset"); applyState(r.result); await refreshImage(); await refreshDistribution(); refreshVersions(); }
    catch (e) { reportError(e); }
  };
  $("moveBtn").onclick = async () => {
    try {
      showError(null);
      const r = await post("move_light");
      applyState(r.result);
      await refreshImage();
      await refreshDistribution();
    } catch (e) { reportError(e); }
  };
  $("saveVerBtn").onclick = async () => {
    try {
      const label = prompt("给这个分布版本加个标签：", "inspect") || "ui";
      const r = await post("save_version", { label });
      applyState(r.result);
      refreshVersions();
    } catch (e) { reportError(e); }
  };
  $("renderImg").addEventListener("click", (ev) => {
    const rect = ev.target.getBoundingClientRect();
    const k = state.kernel;
    const px = Math.floor(((ev.clientX - rect.left) / rect.width) * k.width);
    const py = Math.floor(((ev.clientY - rect.top) / rect.height) * k.height);
    state.probe = { px, py };
    refreshDistribution();
  });
  $("loadVerBtn").onclick = async () => {
    const fn = $("versionSelect").value;
    if (!fn) return;
    state.viewVersion = fn;
    try {
      await post("load_version", { filename: fn });
      await refreshDistribution();
    } catch (e) {
      state.viewVersion = null;
      reportError(e);
      try { await refreshDistribution(); } catch (_) { /* noop */ }
    }
  };
  $("liveVerBtn").onclick = () => {
    state.viewVersion = null;
    refreshDistribution();
  };
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    ["fixed", "move", "acceptance", "coverage", "files"].forEach((name) =>
      $("tab-" + name).classList.toggle("hidden", name !== t.dataset.tab));
  });
  window.addEventListener("resize", () => state.kernel && drawCharts(state.kernel));
}

/* ---------------------------------------------------------------- init */
(async function init() {
  bindStatic();
  try {
    const schema = await api("/api/schema");
    buildSchemaUI(schema);
    buildRegressionUI(schema);
    const st = await api("/api/state");
    applyState(st.state);
    state.probe = { px: Math.floor(st.state.width * 0.5), py: Math.floor(st.state.height * 0.62) };
    await refreshImage();
    await refreshDistribution();
    await refreshVersions();
    await loadResults();
    await loadAcceptance();
    // Gentle state sync in case another tab drives the kernel.
    setInterval(async () => {
      if (state.busy || state.playing) return;
      try { const st = await api("/api/state"); applyState(st.state); } catch (e) {}
    }, 4000);
  } catch (e) {
    showError("界面初始化失败：" + e.message + "。请确认通过 python3 run_ui.py 启动。");
  }
})();
