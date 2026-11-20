// 首屏载荷未到时的兜底；平时以 /api/workspaces 下发的 doc_type_labels 为准。
const SHIPPED_DOC_LABELS = {
  BUSINESS_LICENSE: "营业执照",
  BANK_STATEMENT: "银行流水",
  INCOME_STATEMENT: "利润表",
  CREDIT_REPORT_ENTERPRISE: "企业征信报告",
  VAT_INVOICE: "增值税发票",
  CUSTOMS_VAT_PAYMENT: "海关进口增值税专用缴款书",
  UNKNOWN: "未知类型",
};

const REASON_LABELS = {
  GROUNDING_MISSING: "坐标缺失",
  LOW_OCR: "OCR 置信度低",
  FORMAT_FAIL: "格式校验失败",
  CROSS_FIELD_FAIL: "跨字段校验失败",
  EXTRACTOR_CONFLICT: "抽取通路冲突",
  RAW_UNVERIFIED: "原文在页面上找不到",
  PLACEHOLDER_EMPTY: "占位符已折成空值",
  ENUM_UNKNOWN: "原文就写「未知」",
  REQUIRED_EMPTY: "必填为空",
  AMBIGUOUS_UNIT: "单位不明",
  AMBIGUOUS_DATE: "日期不明",
  UNKNOWN_DOC_TYPE: "未知文档类型",
  PARSE_LOW_QUALITY: "解析质量低",
  NON_CNY_CURRENCY: "非人民币",
  TEMPLATE_MISMATCH: "模板不匹配",
  CHAIN_UNRELIABLE: "流水链条不可靠",
  SIGN_INFERRED: "符号由方向推断",
  LOW_CONFIDENCE: "置信度未达阈值",
  MISSING_PREDICTION: "缺少预测字段",
  SEAL_OVERLAP: "印章压住字段",
  TABLE_ANOMALY: "表格疑似合并单元格、列错位或横向并框",
  QR_CONFLICT: "票面二维码与 OCR 读数冲突",
  LOW_RESOLUTION: "页面分辨率偏低",
  LINE_FALLBACK: "无表头利润表按行读取",
  MATRIX_FALLBACK: "重复表头矩阵按位置读取",
  LOCAL_RECROP: "标签右侧裁切补读",
};

const PHOTO_LABELS = {
  LOW_RESOLUTION: "页面分辨率偏低",
  PHOTO_BLUR: "照片偏糊",
  PHOTO_DARK: "光线偏暗",
  PHOTO_WASHED: "反光或字迹太浅",
  PHOTO_SKEW_UNCORRECTED: "倾斜过大，未能拉正",
  PHOTO_PARTIAL: "只拍到部分页面",
  PHOTO_LOW_OCR: "文字识别置信度偏低",
  PHOTO_OCCLUSION: "疑似手指或物体遮挡",
};

const BASIS_LABELS = {
  calibrated: "已校准",
  calibrated_low_samples: "样本偏少 · 只能排序",
  heuristic: "未校准 · 只能排序",
};

// 复核动作词以 /api/workspaces 下发的 review_vocab 为准，这里只兜首屏。
const SHIPPED_REVIEW_VOCAB = { confirm: "accept", correct: "correct" };

function reviewWords() {
  const vocab = state.reviewVocab || {};
  return {
    confirm: vocab.confirm_action || SHIPPED_REVIEW_VOCAB.confirm,
    correct: vocab.correct_action || SHIPPED_REVIEW_VOCAB.correct,
  };
}

const state = {
  summary: null,
  documents: [],
  workspace: null,
  workspaces: [],
  docId: null,
  doc: null,
  reviews: {},
  ledgerStale: 0,
  tab: "review",
  search: "",
  selected: null,
  correcting: null,
  correctionDraft: "",
  correctionInitial: "",
  switchingWorkspace: false,
  page: 1,
  zoom: 100,
  visiblePaths: [],
  pendingFiles: [],
  fileResults: new Map(),
  importing: false,
  reviewing: false,
  documentRequest: 0,
  allowedSuffixes: [],
  maxUploadMb: 60,
  maxImportFiles: 200,
  reviewVocab: null,
  docLabels: null,
  hintDefault: "",
};

const $ = (id) => document.getElementById(id);

function hasDirtyCorrection() {
  return state.correcting !== null && state.correctionDraft !== state.correctionInitial;
}

function confirmDiscardCorrection() {
  return !hasDirtyCorrection() || window.confirm("修正内容尚未保存，确定放弃这些修改吗？");
}

function esc(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtInt(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function fmtPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function fmtSize(bytes) {
  const size = Number(bytes || 0);
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${size} B`;
}

function fmtValue(value) {
  if (value === null || value === undefined || value === "") return { text: "（空）", empty: true };
  return { text: String(value), empty: false };
}

function confClass(confidence) {
  const value = Number(confidence || 0);
  if (value >= 0.97) return "high";
  if (value >= 0.93) return "mid";
  return "low";
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => el.classList.remove("show"), 2200);
}

function isReviewed(path) {
  return Boolean(state.reviews[path]);
}

function pendingFields() {
  if (!state.doc) return [];
  return Object.values(state.doc.fields).filter(
    (field) => field.decision === "review" && !isReviewed(field.path)
  );
}

function doneFields() {
  if (!state.doc) return [];
  return Object.values(state.doc.fields).filter((field) => isReviewed(field.path));
}

function allFields() {
  return state.doc ? Object.values(state.doc.fields) : [];
}

function docTypeLabel(type) {
  const labels = state.docLabels || {};
  return labels[type] || SHIPPED_DOC_LABELS[type] || type;
}

function renderImportFailures() {
  const failures = state.summary?.failures || [];
  $("import-warning").hidden = !failures.length;
  $("import-warning-text").textContent = `${failures.length} 份资料解析失败，未计入结果；`
    + (state.documents.length ? `其余 ${state.documents.length} 份仍可复核。` : "当前没有可复核的资料，请查看失败详情。");
  $("import-failures").hidden = !failures.length;
  $("import-failures").innerHTML = failures.length ? `<p><strong>${failures.length} 份解析失败，未计入结果</strong><br>${
    state.workspace === "user"
      ? "原件仍在本机 inbox/。请检查或修复原件后重试解析；无需重新上传，也可关闭窗口继续复核正常资料。"
      : "请检查当前数据区的原件并重新运行解析；此处导入的新文件将写入「我的资料」。"}</p><ul>${
    failures.map((item) => `<li>${esc(item.file_name)}<span class="error">${esc(item.error)}</span></li>`).join("")}</ul>` : "";
}

function canRetryImport() {
  return state.workspace === "user" && Boolean(state.summary?.failures?.length);
}

function renderRunStats() {
  renderImportFailures();
  if (!state.summary) return;
  const overall = state.summary.overall || {};
  const calibration = state.summary.calibration || {};
  const requiredKnown = typeof overall.required_field_count === "number";
  const stats = [
    ["文档", fmtInt(state.summary.run && state.summary.run.document_count)],
    ["字段", fmtInt(overall.field_count)],
    [requiredKnown ? "自动接受（必填）" : "自动接受", fmtPct(overall.auto_accept_rate)],
  ];
  if (typeof overall.field_accept_rate === "number") {
    stats.push(["全部字段接受", fmtPct(overall.field_accept_rate)]);
  }
  stats.push(
    ["自动待复核", fmtInt(overall.review_count)],
    ["平均置信度", Number(overall.avg_confidence || 0).toFixed(3)],
  );
  const note = calibration.note
    ? `<div class="stat-note">${esc(calibration.note)}（${fmtInt(calibration.ordinal_count || 0)} / ${
        fmtInt(overall.field_count)} 个字段）</div>`
    : "";
  $("run-stats").innerHTML = stats
    .map(([label, value]) => `<div class="stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`)
    .join("") + note;
}

function renderWorkspaces() {
  const select = $("workspace-select");
  const items = state.workspaces || [];
  if (!items.length) {
    select.innerHTML = `<option value="">当前目录</option>`;
    select.disabled = true;
    return;
  }
  select.disabled = state.reviewing;
  select.innerHTML = items
    .map((item) => `<option value="${esc(item.id)}">${esc(item.label)}${
      item.documents ? ` (${item.documents})` : ""
    }</option>`)
    .join("");
  if (state.workspace) select.value = state.workspace;
}

function docOptionText(id, type, pending) {
  return `${id} · ${docTypeLabel(type)}${pending ? ` · 待复核 ${pending}` : ""}`;
}

function renderDocSelect() {
  const select = $("doc-select");
  if (!state.documents.length) {
    select.innerHTML = `<option value="">暂无文档</option>`;
    select.disabled = true;
    return;
  }
  select.disabled = state.reviewing;
  select.innerHTML = state.documents
    .map((item) => `<option value="${esc(item.document_id)}">${esc(
      docOptionText(item.document_id, item.doc_type, item.review_count))}</option>`)
    .join("");
  if (state.docId) select.value = state.docId;
}

async function activateSummary(summary, workspacePayload, preferredDocId) {
  if (summary) {
    state.summary = summary;
    state.documents = summary.documents || [];
  }
  if (workspacePayload) {
    state.workspace = workspacePayload.current;
    state.workspaces = workspacePayload.items || state.workspaces;
    if (workspacePayload.max_upload_mb) {
      state.maxUploadMb = Number(workspacePayload.max_upload_mb);
    }
    if (workspacePayload.max_import_files) {
      state.maxImportFiles = Number(workspacePayload.max_import_files);
      const label = $("max-files");
      if (label) label.textContent = String(state.maxImportFiles);
    }
    applyIngestTypes(workspacePayload);
  }
  renderRunStats();
  renderDocSelect();
  renderWorkspaces();
  if (!state.documents.length) {
    showEmptyWorkspace();
    return;
  }
  const preferred = preferredDocId && state.documents.some(
    (item) => item.document_id === preferredDocId
  ) ? preferredDocId : null;
  const keep = state.docId && state.documents.some((item) => item.document_id === state.docId)
    ? state.docId : null;
  const first = state.documents.find((item) => item.review_count > 0) || state.documents[0];
  await openDocument(preferred || keep || first.document_id);
}

function showEmptyWorkspace() {
  state.documentRequest += 1;
  $("next-review").disabled = true;
  $("field-search-count").textContent = "0 项";
  state.docId = null;
  state.doc = null;
  state.reviews = {};
  state.selected = null;
  state.visiblePaths = [];
  const label = state.workspace === "user" ? "我的资料" : "当前数据区";
  const failed = state.summary?.failures?.length || 0;
  $("doc-head").innerHTML = `
    <div class="doc-title"><h2>${esc(label)}</h2></div>
    <div class="doc-meta"><span>${failed ? `${failed} 份资料解析失败` : "还没有导入文档"}</span></div>`;
  document.querySelectorAll("#tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === "review");
  });
  $("tab-review-count").textContent = "0";
  $("tab-all-count").textContent = "0";
  $("tab-done-count").textContent = "0";
  $("field-list").innerHTML = failed
    ? `<div class="empty-list">本次没有成功解析的资料。<br>请查看失败详情，检查原件后重试。</div>`
    : `<div class="empty-list">还没有资料。<br>点右上角「导入资料」，选 PDF 或图片即可。</div>`;
  $("page-image").removeAttribute("src");
  $("page-wrap").hidden = true;
  $("overlay").innerHTML = "";
  const empty = $("stage-empty");
  empty.textContent = failed ? "资料解析失败，暂时无法显示原文。" : "导入后在这里看原文。";
  empty.classList.remove("hidden");
  $("page-indicator").textContent = "1 / 1";
  $("prev-page").disabled = true;
  $("next-page").disabled = true;
  $("viewer-hint").textContent = "PDF / 图片 / ZIP 可多选，导入后自动打开第一份。";
  $("page-wrap").style.width = `${state.zoom}%`;
}

function photoChips(quality) {
  const pages = quality.preprocess || [];
  const chips = [];
  const flags = Array.from(new Set(quality.photo_flags || []));
  flags.forEach((flag) => {
    const flagPages = flag === "LOW_RESOLUTION" ? (quality.low_resolution_pages || []) : [];
    const where = flagPages.length ? `（第 ${flagPages.join("、")} 页）` : "";
    const advice = flag === "LOW_RESOLUTION"
      ? "分辨率不足，放大也读不出笔迹，建议重扫"
      : "拍照质量提示，建议重拍";
    chips.push(`<span class="chip bad" title="${esc(advice + where)}">${esc(PHOTO_LABELS[flag] || flag)}</span>`);
  });
  const deskewed = pages.filter((item) => Math.abs(Number(item.deskew_degrees || 0)) >= 0.6);
  if (deskewed.length) {
    const angle = deskewed.reduce(
      (best, item) => (Math.abs(Number(item.deskew_degrees)) > Math.abs(Number(best.deskew_degrees)) ? item : best),
      deskewed[0]);
    // deskew_degrees 带符号，界面显示原来歪了多少度。
    chips.push(`<span class="chip ok" title="原始照片已保留，拉正结果只用于解析与复核">拍歪 ${
      esc(String(Math.abs(Number(angle.deskew_degrees))))}°，已拉正（${deskewed.length} 页）</span>`);
  }
  if (pages.some((item) => item.warped)) {
    chips.push(`<span class="chip ok">透视拉平 ${pages.filter((item) => item.warped).length} 页</span>`);
  }
  const sealed = pages.filter((item) => item.seal_suppressed).length;
  if (sealed) {
    chips.push(`<span class="chip warn" title="章内字迹仍可能被遮挡，压在章上的字段一律进待复核">印章淡化 ${sealed} 页</span>`);
  }
  const fragments = pages.filter((item) => item.seal_edge_fragment).length;
  if (fragments) {
    chips.push(`<span class="chip warn" title="章跨在页面边缘，只拍到半边/一角；两页是否同一枚章尚未核对">`
      + `骑缝章残片 ${fragments} 页</span>`);
  }
  const glare = pages.filter((item) => item.highlight_suppressed).length;
  if (glare) {
    chips.push(`<span class="chip warn" title="只有交给 OCR 的像素被拉伸，原文页仍是拍摄原样">反光压制 ${glare} 页</span>`);
  }
  const lifted = pages.filter((item) => item.contrast_lifted).length;
  if (lifted) {
    chips.push(`<span class="chip warn" title="整页对比度被拉伸之后才交给 OCR（彩色会并成灰度），原文页仍是拍摄原样">`
      + `对比度拉伸 ${lifted} 页</span>`);
  }
  const brightened = pages.filter((item) => item.brightness_lifted).length;
  if (brightened) {
    chips.push(`<span class="chip warn" title="整页按 gamma 提亮之后才交给 OCR，原文页仍是拍摄原样">`
      + `暗页提亮 ${brightened} 页</span>`);
  }
  return chips.join("");
}

function renderDocHead() {
  if (!state.doc) return;
  const pending = pendingFields().length;
  const corrected = doneFields().filter((field) => {
    const record = state.reviews[field.path];
    return record && record.action === reviewWords().correct;
  }).length;
  const autoReview = state.doc.summary.review_count;
  const reviewedPending = doneFields().filter((field) => field.decision === "review").length;
  const quality = state.doc.document_quality || {};
  const parseScore = quality.parse_score;
  const source = state.doc.source;
  const option = $("doc-select").selectedOptions[0];
  if (option) option.textContent = docOptionText(state.docId, state.doc.doc_type, pending);
  const removable = state.workspace === "user";
  $("doc-head").innerHTML = `
    <div class="doc-title">
      <h2>${esc(state.doc.document_id)}</h2>
      <span class="doc-type">${esc(docTypeLabel(state.doc.doc_type))}</span>
      ${removable ? `<button id="doc-remove" class="icon-btn danger" type="button" title="删除这份资料（原件移入 work/trash，不直接销毁）">删除</button>` : ""}
    </div>
    <div class="doc-meta">
      <span>来源 <b>${esc(source.file_name)}</b></span>
      <span>页数 <b>${fmtInt(source.page_count)}</b></span>
      <span>字段 <b>${fmtInt(state.doc.summary.field_count)}</b></span>
      <span>自动待复核 <b>${fmtInt(pending)}</b></span>
      <span>复核进度 <b>${fmtInt(reviewedPending)} / ${fmtInt(autoReview)}</b></span>
      <span>已修正 <b>${fmtInt(corrected)}</b></span>
      <span>解析分 <b>${parseScore == null ? "-" : Number(parseScore).toFixed(3)}</b></span>
    </div>
    <div class="doc-chips">${photoChips(quality)}</div>
    ${state.ledgerStale ? `<div class="doc-refusal">${state.ledgerStale} 条历史复核记录与当前原件不匹配，已保留但不应用；请重新核对相关字段。</div>` : ""}
    ${state.doc.document_refusal ? `<div class="doc-refusal">文档类型未识别（${
      esc(state.doc.document_refusal.code)}）：这份资料没抽出字段，不等于核对过。${
      state.doc.document_refusal.action ? " 怎么办：" + esc(state.doc.document_refusal.action) : ""}</div>` : ""}`;
}

async function removeCurrentDocument() {
  if (!state.docId || state.workspace !== "user" || state.reviewing || state.importing) return;
  const docId = state.docId;
  const next = state.documents.find((item) => item.document_id !== docId);
  const ok = window.confirm(
    `删除「${docId}」？\n\n原件、解析缓存和复核记录会移入 work/trash/，不直接销毁；剩余资料重新跑一遍。`
    + (hasDirtyCorrection() ? "\n\n当前未保存的修正也会丢失。" : ""));
  if (!ok) return;
  try {
    let data;
    setReviewBusy(true);
    try {
      data = await fetchJSON(`/api/document/${encodeURIComponent(docId)}/remove`, { method: "POST" });
    } finally {
      setReviewBusy(false);
    }
    // 删除确认已包含草稿；请求失败时不退出修正。
    state.correcting = null;
    state.docId = null;
    state.doc = null;
    await activateSummary(data.summary, data.workspace, next && next.document_id);
    const pruned = Array.isArray(data.pruned) ? data.pruned.length : 0;
    toast(`已删除 ${docId}，回收位置：${data.trash}`
      + (pruned ? ` · 顺带清理了 ${pruned} 个更早的回收目录` : ""));
  } catch (error) {
    toast(error.message);
  }
}

function renderTabs() {
  const counts = {
    review: pendingFields().length,
    all: allFields().length,
    done: doneFields().length,
  };
  document.querySelectorAll("#tabs button").forEach((button) => {
    const tab = button.dataset.tab;
    button.classList.toggle("active", tab === state.tab);
  });
  $("tab-review-count").textContent = counts.review;
  $("tab-all-count").textContent = counts.all;
  $("tab-done-count").textContent = counts.done;
}

function fieldsForTab() {
  if (state.tab === "review") return pendingFields();
  if (state.tab === "done") return doneFields();
  return allFields();
}

function matchesSearch(field) {
  if (!state.search) return true;
  const haystack = [
    field.label,
    field.path,
    field.value,
    field.raw_text,
    state.reviews[field.path] && state.reviews[field.path].corrected_value,
    (field.review_reasons || []).map((reason) => `${reason} ${REASON_LABELS[reason] || ""}`).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(state.search);
}

function mergeGroups(field) {
  return (field.meta && field.meta.table_anomaly_columns) || [];
}

function reasonChips(field) {
  const groups = mergeGroups(field);
  return (field.review_reasons || [])
    .map((reason) => {
      let title = REASON_LABELS[reason] || reason;
      if (reason === "TABLE_ANOMALY" && groups.length) {
        title += `：${groups.map((cols) => cols.join(" / ")).join("；")}`;
      }
      return `<span class="reason" title="${esc(title)}">${esc(reason)}</span>`;
    })
    .join("");
}

function tableAnomalyRow(field) {
  const groups = mergeGroups(field);
  if (!groups.length) return "";
  const text = groups
    .map((cols) => cols.map((col) => `「${col}」`).join(" + "))
    .join("、");
  return `<div class="detail-row"><span class="k">并框</span><span class="v">${esc(text)} 被读成同一框，值未拆开，请人工分列</span></div>`;
}

function lineFallbackRow(field) {
  if (!(field.meta && field.meta.line_fallback)) return "";
  return `<div class="detail-row"><span class="k">行回落</span><span class="v">没有项目/金额表头，按行名与年份列右对齐取值，请核对列名</span></div>`;
}

function extractorConflictRow(field) {
  const conflict = field.meta && field.meta.extractor_conflict;
  if (!conflict) return "";
  const other = conflict.alt_raw || conflict.alt_value || "空";
  return `<div class="detail-row"><span class="k">双通路</span><span class="v">${esc(conflict.path || "")}：另一条通路读到「${esc(other)}」，规则不一致，值未改，请核对列归属</span></div>`;
}

function rawUnverifiedRow(field) {
  const held = field.meta && field.meta.raw_unverified;
  if (!held) return "";
  return `<div class="detail-row"><span class="k">原文回查</span><span class="v">「${esc(held.raw_text || "")}」在第 ${esc(String(held.page || "-"))} 页找不到，值未改，请核对是否被拼接</span></div>`;
}

// 改写说明由 Python 的 value_rewrite_note 统一生成。
function rewriteRow(field) {
  if (!field.rewrite) return "";
  return `<div class="detail-row"><span class="k">值改写</span><span class="v">${esc(field.rewrite)}</span></div>`;
}

function chainRow(field) {
  if (!field.chain) return "";
  return `<div class="detail-row"><span class="k">链诊断</span><span class="v">${esc(field.chain)}</span></div>`;
}

function reviewActionRow(field) {
  const byReason =
    (state.summary && state.summary.review_actions && state.summary.review_actions.by_reason) || {};
  const actions = [];
  for (const reason of field.review_reasons || []) {
    const action = byReason[reason] && byReason[reason].action;
    if (action && !actions.includes(action)) actions.push(action);
  }
  if (!actions.length) return "";
  return `<div class="detail-row"><span class="k">怎么办</span><span class="v">${esc(actions.join("；"))}</span></div>`;
}

function validatorsHtml(field) {
  const items = field.validators || [];
  if (!items.length) return `<div class="detail-row"><span class="k">校验</span><span class="v">无</span></div>`;
  const rows = items
    .map((item) => {
      const status = item.status === "pass" ? "通过" : item.status === "fail" ? "失败" : "提示";
      return `<li>
        <span class="vid">${esc(item.id)}</span>
        <span class="vstatus ${esc(item.status)}">${status}</span>
        <span class="vdetail">${esc(item.detail || "")}</span>
      </li>`;
    })
    .join("");
  return `<div class="detail-row"><span class="k">校验</span></div>
    <ul class="validators">${rows}</ul>`;
}

function groundingText(field) {
  const grounding = field.grounding;
  if (!grounding || grounding.page == null) return "无坐标";
  const block = grounding.block_id ? ` · 块 ${grounding.block_id}` : "";
  return `P${grounding.page}${block}`;
}

function reviewNote(field) {
  const record = state.reviews[field.path];
  if (!record) return "";
  if (record.action === reviewWords().correct) {
    return `<div class="review-note corrected">
      已修正 <span class="old">${esc(record.original_value ?? "")}</span>
      → <span class="new">${esc(record.corrected_value ?? "")}</span>
      <br>${esc(record.reviewer || "local")} · ${esc(record.timestamp || "")}
    </div>`;
  }
  return `<div class="review-note">
    已通过 · ${esc(record.reviewer || "local")} · ${esc(record.timestamp || "")}
  </div>`;
}

function detailHtml(field) {
  const value = fmtValue(field.value);
  const correcting = state.correcting === field.path;
  const reviewed = isReviewed(field.path);
  return `
    <div class="field-detail">
      <div class="detail-row"><span class="k">原文</span><span class="v">${esc(field.raw_text || "-")}</span></div>
      <div class="detail-row"><span class="k">抽取器</span><span class="v">${esc(field.extractor_id || "-")}</span></div>
      <div class="detail-row"><span class="k">定位</span><span class="v">${esc(groundingText(field))}</span></div>
      ${rewriteRow(field)}
      ${chainRow(field)}
      ${tableAnomalyRow(field)}
      ${lineFallbackRow(field)}
      ${extractorConflictRow(field)}
      ${rawUnverifiedRow(field)}
      ${reviewActionRow(field)}
      <div class="detail-row"><span class="k">类型</span><span class="v">${esc(field.value_type || "-")} · ${esc(field.risk || "-")}${field.required ? " · 必填" : ""}</span></div>
      <div class="detail-row"><span class="k">置信度</span><span class="v${basisInfo(field).ordinal ? " ordinal" : ""}">${esc(basisInfo(field).text)}</span></div>
      ${validatorsHtml(field)}
      ${reviewNote(field)}
      <div class="actions">
        <button class="btn" data-action="prev-item" type="button">上一项</button>
        <button class="btn" data-action="next-item" type="button">下一项</button>
        <button class="btn primary" data-action="${reviewWords().confirm}" type="button">${reviewed ? "再次通过" : "确认通过"}</button>
        <button class="btn warn" data-action="${reviewWords().correct}" type="button">修正</button>
      </div>
      ${
        correcting
          ? `<div class="correct-form">
              <input id="correct-input" type="text" value="${esc(state.correctionDraft)}" placeholder="输入修正后的值">
              <button class="btn primary" data-action="save-correct" type="button">保存</button>
              <button class="btn" data-action="cancel-correct" type="button">取消</button>
            </div>`
          : ""
      }
    </div>`;
}

function basisInfo(field) {
  const basis = field.confidence_basis || "unknown";
  const label = BASIS_LABELS[basis] || basis;
  const samples = field.calibration_samples;
  return {
    ordinal: basis !== "calibrated",
    text: samples == null ? label : `${label} · 校准样本 ${fmtInt(samples)}`,
  };
}

function cardHtml(field) {
  const reviewed = state.reviews[field.path];
  const words = reviewWords();
  const value = reviewed && reviewed.action === words.correct
    ? { text: String(reviewed.corrected_value ?? ""), empty: false }
    : fmtValue(field.value);
  const badge = reviewed
    ? `<span class="badge done">${reviewed.action === words.correct ? "已修正" : "已通过"}</span>`
    : `<span class="badge ${field.decision === "accept" ? "accept" : "review"}">${
        field.decision === "accept" ? "自动接受" : "待复核"
      }</span>`;
  const selected = state.selected === field.path;
  const confidence = Number(field.confidence || 0);
  const basis = basisInfo(field);
  return `
    <article class="field-card ${selected ? "selected" : ""} ${reviewed ? "reviewed" : ""}" data-path="${esc(field.path)}">
      <div class="field-top">
        <div class="field-name">
          <span class="label">${esc(field.label || field.path)}</span>
          <span class="path">${esc(field.path)}</span>
        </div>
        ${badge}
      </div>
      <div class="field-value ${value.empty ? "empty" : ""}">${esc(value.text)}</div>
      <div class="field-meta">
        <span class="conf ${confClass(confidence)}${basis.ordinal ? " ordinal" : ""}" title="${esc(basis.text)}">${confidence.toFixed(3)}${basis.ordinal ? "*" : ""}</span>
        <span class="page-tag">${field.grounding && field.grounding.page ? `P${field.grounding.page}` : "无坐标"}</span>
        ${reasonChips(field)}
      </div>
      ${selected ? detailHtml(field) : ""}
    </article>`;
}

function renderFields() {
  const list = $("field-list");
  const scrollTop = list.scrollTop;
  const fields = fieldsForTab().filter(matchesSearch);
  state.visiblePaths = fields.map((field) => field.path);
  if (state.selected && !state.visiblePaths.includes(state.selected) && !state.correcting) {
    state.selected = null;
  }
  $("field-search-count").textContent = state.search
    ? `匹配 ${fields.length} / ${fieldsForTab().length} 项` : `${fields.length} 项`;
  $("next-review").disabled = state.reviewing || !pendingFields().length;
  if (!fields.length) {
    const refusal = state.doc && state.doc.document_refusal;
    const labels = { review: "没有待复核字段。", all: "没有匹配的字段。", done: "还没有复核记录。" };
    const text = refusal ? "文档类型未识别，这份资料没有抽出字段，不等于核对过。"
      : state.search ? "没有匹配的字段，请更换关键词或清空搜索。" : labels[state.tab];
    list.innerHTML = `<div class="empty-list">${esc(text)}</div>`;
  } else {
    list.innerHTML = fields.map(cardHtml).join("");
  }
  list.scrollTop = scrollTop;
  const selected = list.querySelector(".field-card.selected");
  if (selected) {
    const top = selected.getBoundingClientRect().top - list.getBoundingClientRect().top;
    if (top < 0) list.scrollTop += top;
    else if (top + selected.offsetHeight > list.clientHeight) {
      list.scrollTop += top + Math.min(selected.offsetHeight, list.clientHeight) - list.clientHeight;
    }
  }
  if (state.reviewing) setReviewBusy(true);
}

function pageFields() {
  if (!state.doc) return [];
  return Object.values(state.doc.fields).filter(
    (field) => field.grounding && field.grounding.page === state.page && field.grounding.bbox_norm
  );
}

function pagePreprocess() {
  const quality = (state.doc && state.doc.document_quality) || {};
  const pages = quality.preprocess || [];
  return pages.find((item) => item.page === state.page) || {};
}

function pageSealRegions() {
  return pagePreprocess().seal_regions || [];
}

function renderSealRegions(overlay) {
  const fragment = Boolean(pagePreprocess().seal_edge_fragment);
  for (const region of pageSealRegions()) {
    if (!Array.isArray(region) || region.length !== 4) continue;
    const [x0, y0, x1, y1] = region.map(Number);
    const left = Math.max(0, Math.min(1, x0));
    const top = Math.max(0, Math.min(1, y0));
    const right = Math.max(0, Math.min(1, x1));
    const bottom = Math.max(0, Math.min(1, y1));
    if (!(right > left) || !(bottom > top)) continue;
    const box = document.createElement("div");
    box.className = "seal-box";
    box.title = fragment
      ? "骑缝章残片：只拍到半边/一角，压在残片上的字段都进待复核"
      : "印章区域：章下的字是读出来的，压在章上的字段都进待复核";
    if (fragment) box.dataset.label = "骑缝残片";
    box.style.left = `${left * 100}%`;
    box.style.top = `${top * 100}%`;
    box.style.width = `calc(${(right - left) * 100}% + 1px)`;
    box.style.height = `max(3px, ${(bottom - top) * 100}%)`;
    overlay.appendChild(box);
  }
}

function renderOverlay() {
  const overlay = $("overlay");
  overlay.innerHTML = "";
  if (!state.doc) return;
  renderSealRegions(overlay);
  const compact = window.innerWidth < 720;
  for (const field of pageFields()) {
    const norm = field.grounding.bbox_norm;
    if (!norm || norm.length !== 4) continue;
    // 窄屏只画待复核与选中框，免得流水每格都染色。
    if (compact && field.decision !== "review" && field.path !== state.selected) continue;
    const [x0, y0, x1, y1] = norm;
    const left = Math.max(0, Math.min(1, x0));
    const top = Math.max(0, Math.min(1, y0));
    const right = Math.max(0, Math.min(1, x1));
    const bottom = Math.max(0, Math.min(1, y1));
    if (right <= left || bottom <= top) continue;
    const box = document.createElement("div");
    box.className = "box";
    if (field.decision === "review") box.classList.add("is-review");
    if (field.path === state.selected) box.classList.add("selected");
    box.dataset.path = field.path;
    box.title = `${field.label || field.path}：${field.value == null ? "（空）" : field.value}`;
    box.style.left = `${left * 100}%`;
    box.style.top = `${top * 100}%`;
    box.style.width = `calc(${(right - left) * 100}% + 1px)`;
    box.style.height = `max(3px, ${(bottom - top) * 100}%)`;
    overlay.appendChild(box);
  }
}

function scrollSelectedToCenter() {
  const stage = $("stage");
  const box = $("overlay").querySelector(".box.selected");
  if (!box) return;
  const stageRect = stage.getBoundingClientRect();
  const boxRect = box.getBoundingClientRect();
  stage.scrollLeft += boxRect.left - stageRect.left - stageRect.width / 2 + boxRect.width / 2;
  stage.scrollTop += boxRect.top - stageRect.top - stageRect.height / 2 + boxRect.height / 2;
}

function updatePageChrome() {
  const total = (state.doc && state.doc.source && state.doc.source.page_count) || 1;
  $("page-indicator").textContent = `${state.page} / ${total}`;
  $("prev-page").disabled = state.page <= 1;
  $("next-page").disabled = state.page >= total;
  const field = state.selected && state.doc ? state.doc.fields[state.selected] : null;
  $("viewer-hint").textContent = field
    ? `${field.label || field.path} · ${groundingText(field)}`
    : state.hintDefault;
}

function showPage() {
  if (!state.doc || !state.docId) return;
  const image = $("page-image");
  const empty = $("stage-empty");
  empty.textContent = "正在加载页面…";
  empty.classList.remove("hidden");
  image.onload = () => {
    empty.classList.add("hidden");
    renderOverlay();
    requestAnimationFrame(scrollSelectedToCenter);
  };
  image.onerror = () => {
    empty.textContent = "页面加载失败，检查源文件是否还在。";
    empty.classList.remove("hidden");
  };
  image.src = `/api/page/${encodeURIComponent(state.docId)}/${state.page}.png?dpi=150&t=${Date.now()}`;
  $("page-wrap").hidden = false;
  $("page-wrap").style.width = `${state.zoom}%`;
  renderOverlay();
  updatePageChrome();
}

function selectField(path, jumpToCard) {
  const field = state.doc && state.doc.fields[path];
  if (!field) return;
  if (state.correcting !== path) {
    if (!confirmDiscardCorrection()) return;
    state.correcting = null;
  }
  state.selected = path;
  if (state.search && !matchesSearch(field)) {
    state.search = "";
    $("search").value = "";
  }
  const page = field.grounding && field.grounding.page;
  if (page && page !== state.page) {
    state.page = page;
    showPage();
  }
  renderFields();
  renderOverlay();
  updatePageChrome();
  if (jumpToCard) {
    const card = $("field-list").querySelector(".field-card.selected");
    if (card) card.scrollIntoView({ block: "nearest" });
  } else {
    requestAnimationFrame(scrollSelectedToCenter);
  }
}

function nextPendingPath(afterPath) {
  const fields = allFields();
  const index = fields.findIndex((field) => field.path === afterPath);
  const ordered = fields.slice(index + 1).concat(fields.slice(0, Math.max(0, index + 1)));
  const next = ordered.find((field) => field.decision === "review" && !isReviewed(field.path) && matchesSearch(field));
  return next ? next.path : null;
}

function stepItem(delta) {
  if (!state.visiblePaths.length) return;
  let index = state.visiblePaths.indexOf(state.selected);
  if (index < 0) index = delta > 0 ? -1 : state.visiblePaths.length;
  const next = state.visiblePaths[
    Math.max(0, Math.min(state.visiblePaths.length - 1, index + delta))
  ];
  if (next) selectField(next, true);
}

function advanceAfterReview(path) {
  if (state.tab !== "review") return;
  const next = nextPendingPath(path);
  state.selected = next;
  if (!next) return;
  const field = state.doc.fields[next];
  const page = field.grounding && field.grounding.page;
  if (page && page !== state.page) {
    state.page = page;
    showPage();
  }
}

function jumpToPending() {
  if (state.reviewing) return;
  const next = nextPendingPath(state.selected);
  if (!next) {
    toast(state.search ? "当前搜索下没有待复核字段，可清空搜索查看全部。" : "这份资料已没有待复核字段。");
    return;
  }
  if (state.tab !== "review" || state.correcting !== next) {
    if (!confirmDiscardCorrection()) return;
    state.correcting = null;
  }
  state.tab = "review";
  renderTabs();
  selectField(next, true);
}

function setReviewBusy(busy) {
  state.reviewing = busy;
  $("field-list").setAttribute("aria-busy", String(busy));
  $("field-list").querySelectorAll("[data-action]").forEach((button) => { button.disabled = busy; });
  $("doc-select").disabled = busy || !state.documents.length;
  $("workspace-select").disabled = busy;
  $("search").disabled = busy;
  const input = $("correct-input");
  if (input) input.disabled = busy;
  $("next-review").disabled = busy || !pendingFields().length;
  $("import-open").disabled = busy;
  const remove = $("doc-remove");
  if (remove) remove.disabled = busy;
}

async function postReview(path, action, correctedValue) {
  const body = { field_path: path, action, corrected_value: correctedValue || null };
  const data = await fetchJSON(`/api/review/${encodeURIComponent(state.docId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.reviews[path] = data.record;
  state.correcting = null;
  toast((action === reviewWords().correct ? "已保存修正" : "已确认通过")
        + (data.ledger_unread ? `（${unreadNote(data.ledger_unread)}）` : ""));
}

function unreadNote(count) {
  return `另有 ${count} 行台账读不出来，导出时不会生效`;
}

async function handleAction(action, path) {
  if (state.reviewing || !state.doc || !state.doc.fields[path]) return;
  const words = reviewWords();
  if (action === "prev-item" || action === "next-item") {
    stepItem(action === "next-item" ? 1 : -1);
    return;
  }
  if (action === words.correct) {
    if (state.correcting !== path) {
      if (!confirmDiscardCorrection()) return;
      state.correcting = path;
      const record = state.reviews[path];
      state.correctionDraft = String(record && record.action === words.correct
        ? record.corrected_value : state.doc.fields[path].value ?? "");
      state.correctionInitial = state.correctionDraft;
    }
    renderFields();
    const input = $("correct-input");
    if (input) {
      input.focus();
      input.select();
    }
    return;
  }
  if (action === "cancel-correct") {
    if (!confirmDiscardCorrection()) return;
    state.correcting = null;
    renderFields();
    return;
  }
  if (action !== words.confirm && action !== "save-correct") return;

  const correcting = action === "save-correct";
  let value;
  if (correcting) {
    const input = $("correct-input");
    value = input ? input.value.trim() : "";
    if (!value) {
      toast("修正值不能为空");
      return;
    }
  }
  if (!correcting && !confirmDiscardCorrection()) return;
  setReviewBusy(true);
  try {
    await postReview(path, correcting ? words.correct : words.confirm, value);
    advanceAfterReview(path);
    renderDocHead();
    renderTabs();
    renderFields();
    renderOverlay();
    updatePageChrome();
  } catch (error) {
    toast(error.message);
  } finally {
    setReviewBusy(false);
  }
}

async function openDocument(docId) {
  if ((state.reviewing && !state.switchingWorkspace) || !confirmDiscardCorrection()) {
    $("doc-select").value = state.docId || "";
    return;
  }
  const request = ++state.documentRequest;
  state.docId = docId;
  state.doc = null;
  state.selected = null;
  state.correcting = null;
  state.reviews = {};
  state.search = "";
  $("search").value = "";
  $("doc-head").textContent = "正在加载文档…";
  $("page-wrap").hidden = true;
  $("stage-empty").textContent = "正在加载文档…";
  $("stage-empty").classList.remove("hidden");
  renderTabs();
  renderFields();
  renderOverlay();
  let data;
  try {
    data = await fetchJSON(`/api/document/${encodeURIComponent(docId)}`);
  } catch (error) {
    if (request === state.documentRequest) {
      $("doc-head").textContent = "文档加载失败，请重新选择文档。";
      $("stage-empty").textContent = error.message;
      toast(error.message);
    }
    return;
  }
  if (request !== state.documentRequest) return;
  state.doc = data.document;
  state.ledgerStale = data.ledger_stale || 0;
  for (const record of data.reviews || []) state.reviews[record.field_path] = record;
  if (data.ledger_unread) toast(unreadNote(data.ledger_unread));
  state.tab = pendingFields().length ? "review" : "all";
  const firstPending = pendingFields()[0];
  const first = firstPending || allFields()[0];
  state.page = (first && first.grounding && first.grounding.page) || 1;
  state.selected = first ? first.path : null;
  $("doc-select").value = docId;
  renderDocHead();
  renderTabs();
  renderFields();
  showPage();
}

function fileSuffix(name) {
  const parts = String(name || "").split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

function applyAcceptTypes() {
  const input = $("file-input");
  if (input && state.allowedSuffixes.length) {
    input.setAttribute("accept", state.allowedSuffixes.map((suffix) => "." + suffix).join(","));
  }
}

function applyIngestTypes(payload) {
  if (payload && payload.review_vocab) state.reviewVocab = payload.review_vocab;
  if (payload && payload.doc_type_labels) state.docLabels = payload.doc_type_labels;
  if (!payload || !Array.isArray(payload.allowed_suffixes) || !payload.allowed_suffixes.length) return;
  const suffixes = payload.allowed_suffixes.map((item) => String(item).toLowerCase());
  if (payload.zip_allowed && !suffixes.includes("zip")) suffixes.push("zip");
  state.allowedSuffixes = suffixes;
  applyAcceptTypes();
}

function fileKey(file) {
  const path = file.webkitRelativePath || "";
  return `${path || file.name}:${file.size}:${file.lastModified}`;
}

function addPendingFiles(fileList) {
  if (state.importing) return;
  const maxMb = Number(state.maxUploadMb || 60);
  const maxFiles = Number(state.maxImportFiles || 200);
  const known = new Set(state.pendingFiles.map(fileKey));
  let skipped = 0;
  for (const file of Array.from(fileList || [])) {
    if (state.pendingFiles.length >= maxFiles) {
      toast(`一次最多 ${maxFiles} 个文件，其余已跳过`);
      break;
    }
    const suffix = fileSuffix(file.name);
    if (state.allowedSuffixes.length && !state.allowedSuffixes.includes(suffix)) {
      skipped += 1;
      continue;
    }
    if (file.size > maxMb * 1024 * 1024) {
      toast(`${file.name} 超过 ${maxMb} MB，已跳过`);
      continue;
    }
    const key = fileKey(file);
    if (known.has(key)) continue;
    known.add(key);
    state.pendingFiles.push(file);
  }
  if (skipped) toast(`${skipped} 个文件类型不支持（${state.allowedSuffixes.join(" / ")}），已跳过`);
  renderPendingFiles();
}

function renderPendingFiles() {
  const list = $("import-list");
  list.innerHTML = state.pendingFiles.map((file, index) => {
    const result = state.fileResults.get(fileKey(file));
    return `<li>
      <span class="name" title="${esc(file.webkitRelativePath || file.name)}">${esc(file.name)}</span>
      <span class="size">${fmtSize(file.size)}</span>
      <span class="file-status ${result ? (result.parseFailed ? "error" : esc(result.status)) : ""}" role="status">${esc(result ? result.text : "等待上传")}</span>
      <button class="remove" data-remove="${index}" type="button" title="仅移出上传队列；已上传的原件仍保留在本机" ${state.importing ? "disabled" : ""}>移除</button>
    </li>`;
  }).join("");
  const retry = state.pendingFiles.some((file) => state.fileResults.has(fileKey(file)));
  const retryParsing = canRetryImport() && state.pendingFiles.every(
    (file) => state.fileResults.get(fileKey(file))?.status === "uploaded");
  $("import-start").disabled = state.importing || (!state.pendingFiles.length && !canRetryImport());
  $("import-start").textContent = state.importing ? "导入中…" : retryParsing ? "重试解析" : retry ? "重试导入" : "开始导入";
  renderImportFailures();
  for (const id of ["file-input", "folder-input", "folder-open", "import-close", "import-cancel"]) {
    $(id).disabled = state.importing;
  }
  $("drop-zone").setAttribute("aria-disabled", String(state.importing));
}

function setImportStatus(message, kind) {
  const status = $("import-status");
  status.textContent = message || "";
  status.className = `import-status${kind ? ` ${kind}` : ""}`;
}

function openImportDialog() {
  if (state.reviewing || !confirmDiscardCorrection()) return;
  if (state.correcting) {
    state.correcting = null;
    renderFields();
  }
  setImportStatus("");
  renderPendingFiles();
  const dialog = $("import-dialog");
  if (dialog.showModal) dialog.showModal();
  else dialog.setAttribute("open", "");
}

function closeImportDialog(force = false) {
  if (state.importing && !force) return;
  const dialog = $("import-dialog");
  if (dialog.close) dialog.close();
  else dialog.removeAttribute("open");
}

async function handleImport() {
  if (state.importing || (!state.pendingFiles.length && !canRetryImport())) return;
  const retryExisting = canRetryImport();
  state.importing = true;
  renderPendingFiles();
  const before = new Set(state.documents.map((item) => item.document_id));
  const files = state.pendingFiles.slice();
  const progress = $("import-progress");
  progress.hidden = false;
  progress.max = Math.max(1, files.length);
  progress.value = 0;
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const key = fileKey(file);
      if (state.fileResults.get(key)?.status !== "uploaded") {
        setImportStatus(`正在上传 ${index + 1}/${files.length}：${file.name}`, "busy");
        state.fileResults.set(key, { status: "uploading", text: "正在上传…" });
        renderPendingFiles();
        try {
          const data = await fetchJSON(`/api/import?filename=${encodeURIComponent(file.name)}`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: file,
          });
          const uploaded = Number(data.extracted_count ?? (data.duplicate ? 0 : 1));
          const duplicates = Number(data.duplicates ?? (data.duplicate ? 1 : 0));
          const skipped = Number(data.skipped ?? 0);
          state.fileResults.set(key, {
            status: "uploaded", uploaded, duplicates, skipped,
            names: data.stored_names || [data.name],
            text: duplicates && !uploaded ? "重复文件，已复用" : `已上传 ${uploaded} 份`
              + (skipped ? `，跳过 ${skipped} 份` : ""),
          });
        } catch (error) {
          state.fileResults.set(key, { status: "error", text: `上传失败：${error.message}` });
        }
      }
      progress.value = index + 1;
      renderPendingFiles();
    }
    const ready = files.map((file) => state.fileResults.get(fileKey(file)))
      .filter((result) => result.status === "uploaded");
    if (!ready.length && !retryExisting) {
      setImportStatus("没有文件上传成功，请移除不支持的文件或点击重试导入。", "error");
      return;
    }
    setImportStatus("正在解析本机收件箱；首次处理可能要几十秒…", "busy");
    progress.removeAttribute("value");
    const result = await fetchJSON("/api/import/run", { method: "POST" });
    const newDocs = (result.document_ids || []).filter((id) => !before.has(id));
    const skipped = ready.reduce((total, item) => total + item.skipped, 0);
    await activateSummary(result.summary, result.workspace, newDocs[0]);
    const failures = result.summary.failures || [];
    const failedNames = new Set(failures.map((item) => item.file_name));
    state.pendingFiles = files.filter((file) => {
      const uploaded = state.fileResults.get(fileKey(file));
      if (uploaded.status === "error") return true;
      uploaded.parseFailed = uploaded.names.some((name) => failedNames.has(name));
      if (uploaded.parseFailed) uploaded.text = "已上传，解析失败；可直接重试";
      return uploaded.parseFailed;
    });
    for (const file of files) {
      if (!state.pendingFiles.includes(file)) state.fileResults.delete(fileKey(file));
    }
    const uploadFailures = state.pendingFiles.filter(
      (file) => state.fileResults.get(fileKey(file)).status === "error").length;
    const message = `当前 ${state.documents.length} 份可复核，${failures.length} 份解析失败`
      + (skipped ? `；跳过 ${skipped} 份不支持的文件` : "")
      + (uploadFailures ? `；${uploadFailures} 个文件上传失败，可移除或重试。` : "");
    if (state.pendingFiles.length || failures.length) setImportStatus(message, "error");
    else closeImportDialog(true);
    toast(message);
  } catch (error) {
    setImportStatus(`解析失败：${error.message}；已上传文件保留，重试时不重复上传。`, "error");
    toast(error.message);
  } finally {
    progress.value = progress.max;
    state.importing = false;
    renderPendingFiles();
  }
}

async function exportResults() {
  if (!state.documents.length) {
    toast("当前数据区没有资料可导出。");
    return;
  }
  const button = $("export-run");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "导出中…";
  try {
    const data = await fetchJSON("/api/export", { method: "POST" });
    const link = document.createElement("a");
    link.href = data.download;
    link.download = data.name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    toast(`已导出 ${data.documents} 份 / ${fmtInt(data.fields)} 个字段\n`
      + (data.failed_count ? `另有 ${data.failed_count} 份解析失败，未计入结果，详情已附在导出包中。\n` : "")
      + `网页：${data.report && data.report.path ? data.report.path : data.dir}\n`
      + `ZIP：${data.path}`);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function openReport() {
  const button = $("report-open");
  if (!state.documents.length) {
    toast("当前没有资料。上方「数据」可切到「示例数据」，也可导入自己的 PDF。");
    return;
  }
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "生成中…";
  try {
    await fetchJSON("/api/report");
    window.open("/report", "_blank");
  } catch (error) {
    toast(`${error.message}（当前有 ${state.documents.length} 份资料）`);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

async function switchWorkspace(id) {
  if (!id || id === state.workspace || state.reviewing || state.importing || !confirmDiscardCorrection()) {
    $("workspace-select").value = state.workspace || "";
    return;
  }
  state.switchingWorkspace = true;
  setReviewBusy(true);
  state.documentRequest += 1;
  state.doc = null;
  state.selected = null;
  state.correcting = null;
  $("page-wrap").hidden = true;
  $("stage-empty").textContent = "正在切换数据区…";
  $("stage-empty").classList.remove("hidden");
  renderTabs();
  renderFields();
  renderOverlay();
  try {
    const data = await fetchJSON("/api/workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    state.docId = null;
    await activateSummary(data.summary, data.workspace);
  } catch (error) {
    toast(error.message);
    renderWorkspaces();
    $("stage-empty").textContent = "切换失败，请刷新页面后重试。";
  } finally {
    state.switchingWorkspace = false;
    setReviewBusy(false);
  }
}

async function readEntries(reader) {
  // readEntries 分批返回，目录要读到空批次为止。
  const all = [];
  for (;;) {
    const batch = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
    if (!batch.length) return all;
    all.push(...batch);
  }
}

async function filesFromDataTransfer(transfer) {
  const items = transfer && transfer.items ? Array.from(transfer.items) : [];
  const entries = items
    .filter((item) => item.kind === "file")
    .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (!entries.some((entry) => entry.isDirectory)) {
    return Array.from((transfer && transfer.files) || []);
  }
  const files = [];
  const queue = entries.slice();
  while (queue.length) {
    const entry = queue.shift();
    if (entry.isDirectory) {
      queue.push(...(await readEntries(entry.createReader())));
      continue;
    }
    const file = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
    if (file) files.push(file);
  }
  return files;
}

function bindEvents() {
  window.addEventListener("beforeunload", (event) => {
    if (!hasDirtyCorrection() && !state.reviewing) return;
    event.preventDefault();
    event.returnValue = "";
  });

  $("doc-select").addEventListener("change", (event) => openDocument(event.target.value));

  $("workspace-select").addEventListener("change", (event) => switchWorkspace(event.target.value));
  $("import-open").addEventListener("click", openImportDialog);
  $("import-failures-open").addEventListener("click", openImportDialog);
  $("export-run").addEventListener("click", exportResults);
  $("report-open").addEventListener("click", openReport);
  $("import-close").addEventListener("click", () => closeImportDialog());
  $("import-cancel").addEventListener("click", () => closeImportDialog());
  $("import-start").addEventListener("click", handleImport);
  $("import-dialog").addEventListener("cancel", (event) => {
    if (state.importing) event.preventDefault();
  });
  $("next-review").addEventListener("click", jumpToPending);
  $("file-input").addEventListener("change", (event) => {
    addPendingFiles(event.target.files);
    event.target.value = "";
  });
  $("folder-open").addEventListener("click", (event) => {
    event.preventDefault();
    $("folder-input").click();
  });
  $("folder-input").addEventListener("change", (event) => {
    addPendingFiles(event.target.files);
    event.target.value = "";
  });
  $("import-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove]");
    if (!button || state.importing) return;
    const file = state.pendingFiles[Number(button.dataset.remove)];
    state.fileResults.delete(fileKey(file));
    state.pendingFiles.splice(Number(button.dataset.remove), 1);
    renderPendingFiles();
  });
  const dropZone = $("drop-zone");
  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    const transfer = event.dataTransfer;
    if (!transfer) return;
    if (transfer.items && transfer.items.length && transfer.items[0].kind === "file"
        && transfer.items[0].webkitGetAsEntry) {
      filesFromDataTransfer(transfer).then(addPendingFiles, (error) => toast(error.message));
      return;
    }
    addPendingFiles(transfer.files);
  });

  $("doc-head").addEventListener("click", (event) => {
    if (event.target.closest("#doc-remove")) removeCurrentDocument();
  });

  document.querySelectorAll("#tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.reviewing || state.tab === button.dataset.tab || !confirmDiscardCorrection()) return;
      state.tab = button.dataset.tab;
      state.correcting = null;
      renderTabs();
      renderFields();
      renderOverlay();
    });
  });

  $("search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderFields();
    renderOverlay();
  });

  $("prev-page").addEventListener("click", () => {
    if (state.page > 1) {
      state.page -= 1;
      showPage();
    }
  });
  $("next-page").addEventListener("click", () => {
    const total = (state.doc && state.doc.source.page_count) || 1;
    if (state.page < total) {
      state.page += 1;
      showPage();
    }
  });

  $("zoom").addEventListener("input", (event) => {
    state.zoom = Number(event.target.value);
    $("zoom-label").textContent = `${state.zoom}%`;
    $("page-wrap").style.width = `${state.zoom}%`;
    requestAnimationFrame(scrollSelectedToCenter);
  });

  const list = $("field-list");
  list.addEventListener("click", (event) => {
    if (state.reviewing) return;
    const card = event.target.closest(".field-card");
    if (!card) return;
    const action = event.target.closest("[data-action]");
    if (action) {
      event.stopPropagation();
      handleAction(action.dataset.action, card.dataset.path);
      return;
    }
    if (event.target.closest(".field-detail")) return;
    selectField(card.dataset.path, true);
  });

  list.addEventListener("input", (event) => {
    if (event.target.id === "correct-input") state.correctionDraft = event.target.value;
  });

  list.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.id === "correct-input" && !event.isComposing) {
      event.preventDefault();
      event.stopPropagation();
      const card = event.target.closest(".field-card");
      if (card) handleAction("save-correct", card.dataset.path);
    }
  });

  $("overlay").addEventListener("click", (event) => {
    if (state.reviewing) return;
    const box = event.target.closest(".box");
    if (box) selectField(box.dataset.path, true);
  });

  document.addEventListener("keydown", (event) => {
    if (event.isComposing || event.repeat || state.reviewing || $("import-dialog").open) return;
    if (event.key === "Escape" && state.correcting) {
      event.preventDefault();
      handleAction("cancel-correct", state.correcting);
      return;
    }
    const editable = event.target.closest("input, select, textarea, [contenteditable='true']");
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.altKey) {
      if (!state.selected || !state.visiblePaths.includes(state.selected)) return;
      if (editable && event.target.id !== "correct-input") return;
      event.preventDefault();
      handleAction(state.correcting ? "save-correct" : reviewWords().confirm, state.selected);
      return;
    }
    if (editable || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "/") {
      event.preventDefault();
      $("search").focus();
      return;
    }
    if (event.key.toLowerCase() === "n") {
      event.preventDefault();
      jumpToPending();
      return;
    }
    if (event.key.toLowerCase() === "e" && state.selected) {
      event.preventDefault();
      handleAction(reviewWords().correct, state.selected);
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      stepItem(event.key === "ArrowDown" ? 1 : -1);
    }
  });
}

async function init() {
  state.hintDefault = $("viewer-hint").textContent;
  try {
    const [summary, workspaces] = await Promise.all([
      fetchJSON("/api/summary"),
      fetchJSON("/api/workspaces"),
    ]);
    state.summary = summary;
    state.documents = summary.documents || [];
    state.workspaces = workspaces.items || [];
    state.workspace = workspaces.current;
    state.maxUploadMb = Number(workspaces.max_upload_mb || 60);
    applyIngestTypes(workspaces);
    renderRunStats();
    renderDocSelect();
    renderWorkspaces();
    renderPendingFiles();
    bindEvents();
    if (!state.documents.length) {
      showEmptyWorkspace();
      return;
    }
    const first = state.documents.find((item) => item.review_count > 0) || state.documents[0];
    await openDocument(first.document_id);
  } catch (error) {
    $("stage-empty").textContent = error.message;
    $("stage-empty").classList.remove("hidden");
  }
}

init();
