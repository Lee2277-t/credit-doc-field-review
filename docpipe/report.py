"""Build a single-file, offline HTML report that non-technical users can open.

The report is the human-facing deliverable: documents grouped by type, plain
Chinese section titles, corrected values shown as the final value, and the
source page rendered inline so a field can be traced with one click.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .export import load_final_documents
from .preprocess import FLAG_LABELS
from .review_actions import field_actions, review_action_table
from .reviews import action_vocab, payload_vocab
from .textutil import now_iso

ARRAY_PATH = re.compile(r"^(?P<array>[a-zA-Z0-9_.]+)\[(?P<index>\d+)\]\.(?P<column>[a-zA-Z0-9_]+)$")
# schema 里的路径写成 `bs.transactions[i].amount`，结果里的写成 `[0]`：同一个列，两种下标写法。
SPEC_ARRAY_PATH = re.compile(
    r"^(?P<array>[a-zA-Z0-9_.]+)\[(?P<index>[a-zA-Z0-9_]*)\]\.(?P<column>[a-zA-Z0-9_]+)$")

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --ink: #1d2329; --ink-soft: #5b6670; --ink-faint: #8b959d;
  --line: #e3ded3; --line-strong: #cfc8ba; --bg: #f5f2ea; --card: #fffdf8;
  --accent: #0f6b5c; --accent-soft: #e2f0ec; --review: #a5620a; --review-soft: #fbeeda;
  --corrected: #2f6db3; --corrected-soft: #e6effa; --danger: #a3341f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-size: 14px;
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card);
  border-bottom: 1px solid var(--line); padding: 12px 20px; display: flex; flex-wrap: wrap;
  gap: 14px 18px; align-items: center; }
header h1 { font-size: 17px; margin: 0 12px 0 0; }
header .sub { color: var(--ink-faint); font-size: 12px; }
.stats { display: flex; gap: 16px; flex-wrap: wrap; }
.stat b { display: block; font-size: 17px; }
.stat span { font-size: 11.5px; color: var(--ink-soft); }
.toolbar { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
input[type=search], select { padding: 7px 9px; border: 1px solid var(--line-strong);
  border-radius: 8px; font: inherit; background: #fff; color: var(--ink); }
input[type=search] { min-width: 210px; }
.chip { border: 1px solid var(--line-strong); background: #fff; border-radius: 999px;
  padding: 5px 11px; font: inherit; font-size: 12.5px; cursor: pointer; color: var(--ink-soft); }
.chip.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn { padding: 7px 13px; border-radius: 8px; border: 1px solid var(--line-strong);
  background: #fff; font: inherit; font-size: 12.5px; cursor: pointer; }
.layout { display: grid; grid-template-columns: 236px minmax(0, 1fr); gap: 18px; padding: 18px 20px 60px; }
nav { position: sticky; top: 92px; align-self: start; max-height: calc(100vh - 110px);
  overflow: auto; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
nav h2 { font-size: 12px; margin: 10px 6px 6px; color: var(--ink-faint); font-weight: 600; }
nav a { display: flex; justify-content: space-between; gap: 8px; text-decoration: none;
  color: var(--ink); padding: 6px 8px; border-radius: 7px; font-size: 12.5px; }
nav a:hover { background: var(--accent-soft); }
nav a em { font-style: normal; color: var(--ink-faint); font-size: 11.5px; }
main { min-width: 0; display: flex; flex-direction: column; gap: 18px; }
.doc { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.doc-head { padding: 14px 18px; border-bottom: 1px solid var(--line); display: flex;
  flex-wrap: wrap; gap: 8px 18px; align-items: baseline; }
.doc-head h2 { margin: 0; font-size: 16px; }
.badge { font-size: 11.5px; padding: 3px 9px; border-radius: 999px; background: var(--accent-soft);
  color: var(--accent); border: 1px solid rgba(15,107,92,.2); }
.badge.review { background: var(--review-soft); color: var(--review); border-color: rgba(165,98,10,.25); }
.badge.bad { background: #fbe9e7; color: var(--danger); border-color: rgba(176,48,48,.25); }
.note { flex-basis: 100%; font-size: 11.5px; color: var(--review); background: var(--review-soft);
  border: 1px solid rgba(165,98,10,.25); border-radius: 8px; padding: 4px 10px; }
.doc-meta { color: var(--ink-faint); font-size: 12px; }
.doc-refusal { margin-top: 8px; padding: 8px 10px; border: 1px solid #f0c48a; border-radius: 8px;
  background: #fff4e5; color: var(--ink); font-size: 12.5px; }
.doc-body { padding: 6px 18px 18px; display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 18px; }
.doc-body.no-pages { grid-template-columns: minmax(0,1fr); }
section.group { margin-top: 14px; }
section.group > h3 { margin: 0 0 6px; font-size: 13px; color: var(--ink-soft);
  border-left: 3px solid var(--accent); padding-left: 8px; }
table.fields { width: 100%; border-collapse: collapse; font-size: 13px; }
table.fields th, table.fields td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }
table.fields th { font-size: 11.5px; color: var(--ink-faint); font-weight: 600; background: #faf8f3; }
table.fields th:nth-child(1) { width: 38%; }
table.fields th:nth-child(2) { width: auto; }
table.fields th:nth-child(3), table.fields th:nth-child(4),
table.fields th:nth-child(5), table.fields th:nth-child(6) { width: 1%; white-space: nowrap; }
table.fields td:nth-child(n+3) { white-space: nowrap; }
table.fields td:nth-child(5), table.fields td:nth-child(6) { color: var(--ink-soft); font-size: 12px; }
tr.field { cursor: pointer; }
tr.field:hover { background: #f7f4ec; }
tr.field.selected { background: var(--accent-soft); }
tr.field.pending td.value { color: var(--review); }
td.value { font-weight: 600; word-break: break-all; }
td.value .orig { display: block; font-weight: 400; font-size: 11.5px; color: var(--ink-faint);
  text-decoration: line-through; }
.state { font-size: 11.5px; padding: 2px 7px; border-radius: 999px; background: #eef1f0; color: var(--ink-soft); }
.state.review { background: var(--review-soft); color: var(--review); }
.state.corrected { background: var(--corrected-soft); color: var(--corrected); }
.state.accepted { background: var(--accent-soft); color: var(--accent); }
table.rows { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 4px; }
table.rows th, table.rows td { border: 1px solid var(--line); padding: 5px 7px; text-align: left; }
table.rows th { background: #faf8f3; font-size: 11.5px; color: var(--ink-soft); position: sticky; top: 0; }
.table-scroll { max-height: 460px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
tr.hint .why { color: var(--review); font-weight: 600; font-size: 11.5px; margin-right: 7px; }
tr.hint .why-text { color: #6b4a12; font-size: 12px; }
table.fields tr.hint td { border-bottom: 1px solid var(--line); padding: 0 8px 8px;
  background: var(--review-soft); }
table.rows tr.hint td { border: 0; border-bottom: 1px solid var(--line); padding: 6px 8px;
  background: var(--review-soft); }
.pages { border-left: 1px dashed var(--line-strong); padding-left: 16px; }
.pages h3 { font-size: 13px; margin: 14px 0 8px; color: var(--ink-soft); }
.page { position: relative; margin-bottom: 10px; border: 1px solid var(--line-strong); background: #fff; }
.page img { display: block; width: 100%; }
.page .mark { position: absolute; border: 2px solid var(--accent); background: rgba(15,107,92,.18);
  border-radius: 2px; display: none; box-shadow: 0 0 0 3px rgba(15,107,92,.18); }
.page .mark.show { display: block; }
.page .tag { position: absolute; left: 0; top: 0; transform: translateY(-100%); background: var(--accent);
  color: #fff; font-size: 11px; padding: 2px 7px; border-radius: 5px 5px 0 0; display: none; }
.page .tag.show { display: block; }
.empty { color: var(--ink-faint); padding: 30px; text-align: center; }
footer { padding: 0 20px 40px; color: var(--ink-faint); font-size: 11.5px; }
@media (max-width: 980px) {
  .layout { grid-template-columns: minmax(0,1fr); }
  nav { position: static; max-height: none; }
  .doc-body { grid-template-columns: minmax(0,1fr); }
  .pages { border-left: 0; padding-left: 0; }
}
@media (max-width: 760px) {
  table.fields thead { display: none; }
  table.fields, table.fields tbody, tr.field, table.fields td { display: block; width: auto; }
  tr.field { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 10px;
    border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; margin-bottom: 8px; background: #fff; }
  tr.field td { border: 0; padding: 0; white-space: normal; }
  table.fields td:nth-child(1) { flex: 1 1 100%; order: 2; font-size: 14.5px; }
  table.fields td:nth-child(2) { order: 1; color: var(--ink-soft); font-size: 12.5px; font-weight: 600; }
  table.fields td:nth-child(4) { order: 3; }
  table.fields td:nth-child(3), table.fields td:nth-child(5), table.fields td:nth-child(6) {
    order: 4; color: var(--ink-faint); font-size: 11.5px; }
  table.fields tr.hint { display: block; margin: -4px 0 8px;
    border: 1px solid rgba(165,98,10,.25); border-radius: 9px; }
  table.fields tr.hint td { padding: 8px 11px; font-size: 12px; color: #6b4a12; }
}
@media print {
  header, nav, .toolbar { position: static; }
  .layout { grid-template-columns: minmax(0,1fr); }
  .table-scroll { max-height: none; }
  .doc { break-inside: avoid; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1 id="report-title">__TITLE__</h1>
    <div class="sub" id="report-sub"></div>
  </div>
  <div class="stats" id="report-stats"></div>
  <div class="toolbar">
    <input id="search" type="search" placeholder="搜索字段名 / 值 / 文档">
    <button class="chip active" data-filter="all" type="button">全部</button>
    <button class="chip" data-filter="pending" type="button">待复核</button>
    <button class="chip" data-filter="corrected" type="button">人工修正</button>
    <button class="chip" data-filter="reviewed" type="button">人工已复核</button>
    <button class="btn" id="print" type="button">打印 / 存 PDF</button>
  </div>
</header>
<div class="layout">
  <nav id="nav"></nav>
  <main id="main"></main>
</div>
<footer id="footer"></footer>
<script id="report-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const state = { filter: "all", query: "", selected: null };
// 复核状态词由 Python 按 config 注入，模板不另存一份。
const VOCAB = DATA.meta.review_vocab || {};
const PHOTO_LABELS = __PHOTO_LABELS__;
const esc = (value) => String(value ?? "").replace(/[&<>"]/g, (ch) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function photoBadges(doc) {
  const out = [];
  (doc.photo_flags || []).forEach((flag) =>
    out.push(`<span class="badge bad" title="建议重扫或重拍">${esc(PHOTO_LABELS[flag] || flag)}</span>`));
  const pages = doc.preprocess || [];
  const deskew = pages.filter((item) => Math.abs(Number(item.deskew_degrees || 0)) >= 0.6);
  if (deskew.length) out.push(`<span class="badge">已自动拉正 ${deskew.length} 页</span>`);
  const warp = pages.filter((item) => item.warped).length;
  if (warp) out.push(`<span class="badge">透视拉平 ${warp} 页</span>`);
  const seal = pages.filter((item) => item.seal_suppressed).length;
  if (seal) out.push(`<span class="badge review">印章淡化 ${seal} 页</span>`);
  const glare = pages.filter((item) => item.highlight_suppressed).length;
  if (glare) out.push(`<span class="badge review">反光压制 ${glare} 页</span>`);
  // 影调改写作用于整页，得让看报表的人知道笔画被拉过。
  const contrast = pages.filter((item) => item.contrast_lifted).length;
  if (contrast) out.push(`<span class="badge review">对比度拉伸 ${contrast} 页</span>`);
  const bright = pages.filter((item) => item.brightness_lifted).length;
  if (bright) out.push(`<span class="badge review">暗页提亮 ${bright} 页</span>`);
  return out.join("");
}
function stateLabel(field) {
  if (field.review_status === VOCAB.corrected_state) return "人工修正";
  if (field.review_status === VOCAB.confirming_state) return "人工确认";
  return field.decision === "review" ? "待复核" : "自动接受";
}

// 改写说明由 Python 的 value_rewrite_note 生成，与复核台共用。
function noteRow(label, notes, span) {
  const list = (Array.isArray(notes) ? notes : [notes]).filter(Boolean);
  if (!list.length) return "";
  return `<tr class="hint"><td colspan="${span}"><span class="why">${label}</span><span class="why-text">${esc(list.join("；"))}</span></td></tr>`;
}
const hintRow = (actions, span) => noteRow("怎么办", actions, span);
const rewriteRow = (notes, span) => noteRow("值改写", notes, span);

// 链诊断是文档级事实，只在文档头部提一次。
function chainNote(doc) {
  if (!doc.chain) return "";
  return `<div class="note">链诊断：${esc(doc.chain)}</div>`;
}

function matches(field) {
  if (state.filter === "pending" && !(field.decision === "review" && !field.review_status)) return false;
  if (state.filter === "corrected" && field.review_status !== VOCAB.corrected_state) return false;
  if (state.filter === "reviewed" && !field.review_status) return false;
  if (!state.query) return true;
  const haystack = `${field.label} ${field.path} ${field.value} ${field.original_value} ${field.doc_label} ${field.doc_id}`.toLowerCase();
  return haystack.includes(state.query);
}

function fieldRow(field) {
  const cls = ["field"];
  if (field.decision === "review" && !field.review_status) cls.push("pending");
  const status = field.review_status === VOCAB.corrected_state ? "corrected"
    : field.review_status ? "accepted" : field.decision === "review" ? "review" : "accepted";
  const original = field.review_status === VOCAB.corrected_state && field.original_value !== field.value
    ? `<span class="orig">自动识别：${esc(field.original_value)}</span>` : "";
  return `<tr class="${cls.join(" ")}" data-doc="${esc(field.doc_id)}" data-path="${esc(field.path)}" data-page="${esc(field.page || "")}">
    <td class="value">${original}${esc(field.value === "" ? "（未识别）" : field.value)}</td>
    <td>${esc(field.label)}</td>
    <td>${esc(field.kind || "-")}</td>
    <td><span class="state ${status}">${esc(stateLabel(field))}</span></td>
    <td>${field.confidence == null ? "-" : esc(field.confidence.toFixed(2)) + (field.ordinal ? " *" : "")}</td>
    <td>${field.page ? "P" + esc(field.page) : "-"}</td>
  </tr>`;
}

function renderDoc(doc) {
  const sections = doc.sections.map((section) => {
    const rows = section.fields.filter((field) => field.doc_id === doc.id && matches(field));
    if (!rows.length) return "";
    return `<section class="group"><h3>${esc(section.title)}（${rows.length}）</h3>
      <table class="fields"><thead><tr><th>值</th><th>字段</th><th>类型</th><th>复核状态</th><th>置信度</th><th>页码</th></tr></thead>
      <tbody>${rows.map((field) => fieldRow(field) + rewriteRow(field.rewrite, 6) + hintRow(field.actions, 6)).join("")}</tbody></table>
      ${rows.some((item) => item.ordinal) ? '<div class="doc-meta">* 置信度只可排序，不能当概率读</div>' : ""}</section>`;
  }).join("");
  const tables = doc.tables.map((table) => {
    const rows = state.query
      ? table.rows.filter((row) => row.cells.some((cell) => String(cell.value).toLowerCase().includes(state.query)))
      : table.rows;
    if (state.query && !rows.length) return "";
    return `
    <section class="group"><h3>${esc(table.title)}（${rows.length} 行）</h3>
      <div class="table-scroll"><table class="rows"><thead><tr>${
        table.columns.map((column) => `<th>${esc(column)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${row.cells.map((cell) => `<td>${esc(cell.value)}${
          cell.corrected ? " *" : cell.pending ? " †" : ""}</td>`).join("")}</tr>${
          rewriteRow(row.rewrites, table.columns.length) +
          hintRow(row.actions, table.columns.length)}`).join("")}
        </tbody></table></div>
      <div class="doc-meta">* 人工修正过 · † 仍待复核</div></section>`;
  }).join("");
  const pages = doc.pages.length ? `<aside class="pages"><h3>原文页</h3>${
    doc.pages.map((page) => `<div class="page" data-page="${page.page}" data-doc="${esc(doc.id)}">
      <span class="tag"></span><img alt="第 ${page.page} 页" src="${page.image}">
      <div class="mark"></div></div>`).join("")}</aside>` : "";
  const bodyClass = doc.pages.length ? "doc-body" : "doc-body no-pages";
  return `<article class="doc" id="doc-${esc(doc.id)}">
    <div class="doc-head">
      <h2>${esc(doc.id)}</h2><span class="badge">${esc(doc.doc_type_label)}</span>
      ${doc.stats.pending_count ? `<span class="badge review">待复核 ${doc.stats.pending_count}</span>` : ""}
      ${doc.stats.corrected_count ? `<span class="badge">人工修正 ${doc.stats.corrected_count}</span>` : ""}
      ${photoBadges(doc)}
      ${chainNote(doc)}
      ${doc.refusal ? `<div class="doc-refusal">文档类型未识别（${esc(doc.refusal.code)}）：这份资料没抽出字段，不等于核对过。${doc.refusal.action ? " 怎么办：" + esc(doc.refusal.action) : ""}</div>` : ""}
      <div class="doc-meta">${esc(doc.file_name)} · ${doc.page_count} 页 · ${doc.stats.field_count} 字段 · 自动接受 ${(doc.stats.accept_count / Math.max(1, doc.stats.field_count) * 100).toFixed(1)}%</div>
    </div>
    <div class="${bodyClass}"><div>${sections}${tables}</div>${pages}</div>
  </article>`;
}

function render() {
  const docs = DATA.documents;
  const nav = DATA.groups.map((group) => {
    const items = docs.filter((doc) => doc.doc_type === group.type);
    if (!items.length) return "";
    return `<h2>${esc(group.label)}</h2>${items.map((doc) => `<a href="#doc-${esc(doc.id)}">
      <span>${esc(doc.id)}</span><em>${doc.refusal ? "未识别类型" : doc.stats.pending_count ? "待复核 " + doc.stats.pending_count : "已完成"}</em></a>`).join("")}`;
  }).join("");
  document.getElementById("nav").innerHTML = nav || "<div class='empty'>没有匹配的文档</div>";
  // 拒答文档没有字段可筛，得单独放行。
  const visible = docs.filter((doc) => doc.refusal || doc.sections.some((section) =>
    section.fields.some((field) => field.doc_id === doc.id && matches(field))) || doc.tables.length);
  document.getElementById("main").innerHTML = visible.length
    ? visible.map(renderDoc).join("") : "<div class='empty'>没有匹配的字段</div>";
  if (state.selected) highlight(state.selected, false);
}

function renderHead() {
  const overall = DATA.overall;
  const stamp = String(DATA.meta.exported_at || "").replace("T", " ").slice(0, 16);
  document.getElementById("report-sub").textContent =
    `${DATA.meta.workspace_label} 数据区 · 导出于 ${stamp} · 点字段行定位原文`
    + (DATA.meta.pages_note ? ` · ${DATA.meta.pages_note}` : "");
  document.getElementById("report-stats").innerHTML = [
    ["文档", DATA.documents.length], ["字段", overall.field_count],
    ["自动接受", `${(overall.accept_count / Math.max(1, overall.field_count) * 100).toFixed(1)}%`],
    ["待复核", overall.pending_review_count], ["人工修正", overall.reviewed_corrected_count],
  ].map(([label, value]) => `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join("")
    + (DATA.calibration && DATA.calibration.note
      ? `<div class="note">${esc(DATA.calibration.note)}（${DATA.calibration.ordinal_count} / ${overall.field_count} 个字段）</div>` : "")
    + (DATA.failures.length ? `<details class="note"><summary>${DATA.failures.length} 份解析失败，未计入结果</summary><ul>${
        DATA.failures.map((item) => `<li>${esc(item.file_name)}：${esc(item.error)}</li>`).join("")}</ul></details>` : "");
  document.getElementById("footer").textContent =
    `离线可读，不用联网或安装软件。表格数据：final/final_fields.csv；机器可读结果与复核流水在 final/、review/。`;
}

function highlight(target, scroll) {
  document.querySelectorAll(".page .mark, .page .tag").forEach((node) => node.classList.remove("show"));
  document.querySelectorAll("tr.field.selected").forEach((node) => node.classList.remove("selected"));
  if (!target) return;
  const row = findRow(target.doc, target.path);
  if (row) row.classList.add("selected");
  const page = document.querySelector(`.page[data-doc="${cssAttr(target.doc)}"][data-page="${cssAttr(target.page)}"]`);
  if (!page) return;
  const mark = page.querySelector(".mark");
  const tag = page.querySelector(".tag");
  if (!target.bbox) return;
  const [x1, y1, x2, y2] = target.bbox;
  mark.style.left = (x1 * 100) + "%";
  mark.style.top = (y1 * 100) + "%";
  mark.style.width = Math.max(1.5, (x2 - x1) * 100) + "%";
  mark.style.height = Math.max(1.5, (y2 - y1) * 100) + "%";
  mark.classList.add("show");
  tag.textContent = target.label;
  tag.classList.add("show");
  if (scroll !== false) page.scrollIntoView({ behavior: "smooth", block: "center" });
}

function cssAttr(value) {
  return String(value ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function findRow(docId, path) {
  return Array.from(document.querySelectorAll("tr.field")).find(
    (row) => row.dataset.doc === docId && row.dataset.path === path);
}

document.getElementById("main").addEventListener("click", (event) => {
  const row = event.target.closest("tr.field");
  if (!row) return;
  const doc = DATA.documents.find((item) => item.id === row.dataset.doc);
  const section = doc && doc.sections.flatMap((item) => item.fields).find((field) => field.path === row.dataset.path);
  if (!section) return;
  state.selected = { doc: doc.id, path: section.path, page: section.page, bbox: section.bbox_norm, label: section.label };
  highlight(state.selected, true);
});
document.getElementById("search").addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  render();
});
document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
  document.querySelectorAll(".chip").forEach((item) => item.classList.toggle("active", item === chip));
  state.filter = chip.dataset.filter;
  render();
}));
document.getElementById("print").addEventListener("click", () => window.print());
renderHead();
render();
</script>
</body>
</html>
"""


def _page_options(report_config: Dict, override: Optional[Dict]) -> Dict:
    options = {
        "embed_pages": bool(report_config.get("embed_pages", True)),
        "page_dpi": float(report_config.get("page_dpi", 96)),
        "page_jpeg_quality": int(report_config.get("page_jpeg_quality", 62)),
        "max_pages_per_document": int(report_config.get("max_pages_per_document", 8)),
    }
    options.update(override or {})
    return options


def _page_images(source: str, page_count: int, config: Dict,
                 options: Optional[Dict] = None) -> List[Dict]:
    report = _page_options(config.get("report", {}), options)
    if not report["embed_pages"] or not source or not Path(source).is_file():
        return []
    import pymupdf as fitz

    from .parse import open_document

    dpi = report["page_dpi"]
    quality = int(report["page_jpeg_quality"])
    limit = int(report["max_pages_per_document"]) or page_count
    images = []
    doc = open_document(Path(source), config)
    try:
        for index in range(min(page_count, limit)):
            page = doc[index]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            raw = pix.tobytes("jpeg", jpg_quality=quality)
            images.append({
                "page": index + 1,
                "image": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"),
            })
    finally:
        doc.close()
    return images


def _kind_label(field: Dict) -> str:
    types = {"decimal": "金额/数值", "date": "日期", "string": "文本", "enum": "选项", "integer": "数值"}
    return types.get(field.get("value_type"), field.get("value_type") or "文本")


def _field_entry(document_id: str, doc_label: str, path: str, field: Dict,
                 action_table: Dict[str, str]) -> Dict:
    grounding = field.get("grounding") or {}
    return {
        "doc_id": document_id,
        "doc_label": doc_label,
        "path": path,
        "label": field.get("label") or path,
        "value": "" if field.get("effective_value") is None else str(field.get("effective_value")),
        "original_value": "" if field.get("value") is None else str(field.get("value")),
        "kind": _kind_label(field),
        "confidence": float(field.get("confidence") or 0.0),
        "ordinal": str(field.get("confidence_basis") or "unknown") != "calibrated",
        "decision": field.get("decision") or "",
        "review_status": field.get("review_status") or "",
        "page": grounding.get("page"),
        "bbox": grounding.get("bbox"),
        "bbox_norm": grounding.get("bbox_norm"),
        "reasons": field.get("review_reasons") or [],
        "actions": field_actions(field, action_table),
        "rewrite": field.get("rewrite") or "",
    }


def _doc_chain_note(doc: Dict) -> str:
    """链诊断是文档级的一句话，被拦的每一格带的是同一句：取第一次出现的那条。"""
    return next((str(field.get("chain")) for field in doc["fields"].values()
                 if field.get("chain")), "")


def _split_arrays(doc: Dict, config: Dict, action_table: Dict[str, str]) -> tuple:
    """Separate scalar fields from table rows such as bs.transactions[3].amount."""
    spec = config.get("doc_types", {}).get(doc["doc_type"], {})
    arrays: Dict[str, Dict[int, Dict[str, Dict]]] = {}
    scalars: List[Dict] = []
    for path, field in doc["fields"].items():
        match = ARRAY_PATH.match(path)
        if match:
            arrays.setdefault(match.group("array"), {}).setdefault(int(match.group("index")), {})[
                match.group("column")] = field
        else:
            scalars.append(field)
    tables = []
    for array_path, rows in arrays.items():
        table_name = array_path.rsplit(".", 1)[-1]
        # 列的键是英文（`bs.transactions[0].amount`），印在纸上的表头是中文（「交易金额」）。
        # 拿 `tables[*].headers` 当键去取值，一格都取不到——列序与中文名都得从 schema 来。
        spec_columns: List[str] = []
        labels: Dict[str, str] = {}
        for path, field_spec in _field_specs(spec):
            match = SPEC_ARRAY_PATH.match(path)
            if not match or match.group("array") != array_path:
                continue
            spec_columns.append(match.group("column"))
            labels[match.group("column")] = field_spec.get("label", match.group("column"))
        present = {column for row in rows.values() for column in row}
        columns = [column for column in spec_columns if column in present] or sorted(present)
        rendered_rows = []
        corrected_state = action_vocab(config)["corrected_state"]
        for index in sorted(rows):
            cells, actions, rewrites = [], [], []
            for column in columns:
                field = rows[index].get(column)
                if field is None:
                    cells.append({"value": "", "corrected": False, "pending": False})
                    continue
                cells.append({
                    "value": "" if field.get("effective_value") is None else str(field["effective_value"]),
                    "corrected": bool(corrected_state)
                        and field.get("review_status") == corrected_state,
                    "pending": field.get("decision") == "review" and not field.get("review_status"),
                })
                # 一整行只在下面出一次「怎么办」：同一行的哪一格被拦，人要做的是同一件事。
                for text in field_actions(field, action_table):
                    if text not in actions:
                        actions.append(text)
                note = str(field.get("rewrite") or "").strip()
                if note and note not in rewrites:
                    rewrites.append(note)
            rendered_rows.append({"cells": cells, "actions": actions, "rewrites": rewrites})
        title = config.get("report", {}).get("table_titles", {}).get(table_name, table_name)
        tables.append({"title": title, "columns": [labels.get(c, c) for c in columns],
                       "rows": rendered_rows})
    return scalars, tables


def _field_specs(spec: Dict) -> List[tuple]:
    return [(item["path"], item) for item in spec.get("fields", [])]


UNKNOWN_DOC_LABEL = "未知类型"


def doc_type_labels(config: Dict) -> Dict[str, str]:
    """文档类型中文名的唯一主人：配置 `doc_types.*.label`，外加一个未知类型的名字。

    复核台拿不到 YAML，这份表由 `/api/workspaces` 的 `doc_type_labels` 下发；单文件网页版
    在生成时直接嵌进 payload。两处都读这一份，改名只改配置一处。
    """
    labels = {name: (spec or {}).get("label") or name
              for name, spec in (config.get("doc_types") or {}).items()}
    labels.setdefault("UNKNOWN", UNKNOWN_DOC_LABEL)
    return labels


def doc_type_label(doc_type: str, config: Dict) -> str:
    return doc_type_labels(config).get(doc_type) or doc_type or "其他资料"


def build_report(out_dir: Path, dest: Optional[Path] = None,
                 config: Optional[Dict] = None) -> Dict:
    """Write the single-file HTML report into <final>/ and return its location."""
    config = config or load_config()
    report_config = config.get("report", {})
    out_dir = Path(out_dir).resolve()
    final_dir = (dest or out_dir / "final").resolve()
    documents, final_summary, source_summary = load_final_documents(out_dir, config)
    report_vocab = payload_vocab(config)
    report_corrected_state = report_vocab["corrected_state"]

    doc_types = [item["doc_type"] for item in final_summary.get("by_doc_type", [])]
    grouped = list(dict.fromkeys(doc_types))

    payload_docs = []
    page_sources: List[tuple] = []
    mapping = report_config.get("risk_groups") or {}
    fallback_group = report_config.get("fallback_group", "其他信息")
    action_table = review_action_table(config)
    for doc in documents:
        _, tables = _split_arrays(doc, config, action_table)
        label = doc_type_label(doc["doc_type"], config)
        grouped_fields: Dict[str, List[Dict]] = {}
        for path, field in doc["fields"].items():
            if ARRAY_PATH.match(path):
                continue
            group = mapping.get(field.get("risk") or "", fallback_group)
            grouped_fields.setdefault(group, []).append(
                _field_entry(doc["document_id"], label, path, field, action_table))
        order = report_config.get("group_order") or list(grouped_fields)
        sections = [{"title": name, "fields": grouped_fields[name]}
                    for name in [*order, *grouped_fields]
                    if grouped_fields.get(name)]
        seen = set()
        sections = [item for item in sections if not (item["title"] in seen or seen.add(item["title"]))]
        stats = {
            "field_count": len(doc["fields"]),
            "accept_count": sum(1 for f in doc["fields"].values() if f["decision"] == "accept"),
            "review_count": sum(1 for f in doc["fields"].values() if f["decision"] == "review"),
            "reviewed_count": sum(1 for f in doc["fields"].values() if f.get("review_status")),
            "corrected_count": sum(1 for f in doc["fields"].values()
                                   if report_corrected_state
                                   and f.get("review_status") == report_corrected_state),
            "pending_count": sum(1 for f in doc["fields"].values()
                                 if f["decision"] == "review" and not f.get("review_status")),
        }
        payload_docs.append({
            "id": doc["document_id"],
            "doc_type": doc["doc_type"],
            "doc_type_label": label,
            "file_name": (doc.get("source") or {}).get("file_name") or doc["document_id"],
            "page_count": (doc.get("source") or {}).get("page_count") or 1,
            "photo_flags": (doc.get("document_quality") or {}).get("photo_flags") or [],
            "preprocess": (doc.get("document_quality") or {}).get("preprocess") or [],
            "chain": _doc_chain_note(doc),
            **({"refusal": doc["document_refusal"]}
               if doc.get("document_refusal") else {}),
            "sections": sections,
            "tables": tables,
            "pages": [],
            "stats": stats,
        })
        page_sources.append(((doc.get("source") or {}).get("path") or "",
                             int((doc.get("source") or {}).get("page_count") or 0)))

    payload = {
        "meta": {
            "title": report_config.get("title", "信贷资料结构化结果"),
            "workspace_label": out_dir.name,
            "exported_at": final_summary["export"]["exported_at"],
            "pipeline_version": final_summary["export"].get("pipeline_version"),
            "source_out": final_summary["export"]["source_out"],
            "review_vocab": report_vocab,
            "pages_note": "",
        },
        "groups": [{"type": name, "label": doc_type_label(name, config)} for name in grouped],
        "overall": final_summary["overall"],
        "calibration": final_summary.get("calibration") or {},
        "failures": final_summary.get("failures") or [],
        "documents": payload_docs,
    }

    def assemble() -> str:
        return (TEMPLATE
                .replace("__TITLE__", json.dumps(payload["meta"]["title"], ensure_ascii=False).strip('"'))
                # 标签只在 preprocess.FLAG_LABELS 里存一份，报表这边直接注入，免得两处漂移。
                .replace("__PHOTO_LABELS__", json.dumps(FLAG_LABELS, ensure_ascii=False))
                .replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")))

    # 逐级降级：先按配置原样出图，超体积再压质量、少嵌页，最后才整块去掉原文页。
    budget = int(report_config.get("max_size_mb", 12) or 0) * 1024 * 1024
    steps: List[Optional[Dict]] = [None, *(report_config.get("degrade_ladder") or [])]
    level, html, size = 0, "", 0
    for index, step in enumerate(steps):
        for item, (source, page_count) in zip(payload_docs, page_sources):
            item["pages"] = _page_images(source, page_count, config, step)
        options = _page_options(report_config, step)
        payload["meta"]["pages_note"] = "" if index == 0 else (
            "原文页未内嵌（单文件体积上限）" if not options["embed_pages"]
            else f"原文页已压缩（质量 {int(options['page_jpeg_quality'])}、每份最多 "
                 f"{int(options['max_pages_per_document'])} 页）以控制单文件体积")
        html = assemble()
        level = index
        size = len(html.encode("utf-8"))
        if not budget or size <= budget or index == len(steps) - 1:
            break

    name = report_config.get("file_name", "查看结果.html")
    target = final_dir / name
    final_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return {
        "path": str(target),
        "name": name,
        "size": target.stat().st_size,
        "documents": len(payload_docs),
        "fields": final_summary["overall"]["field_count"],
        "embedded_pages": sum(len(doc["pages"]) for doc in payload_docs),
        "page_level": level,
        "max_size_mb": int(report_config.get("max_size_mb", 12) or 0),
    }


__all__ = ["build_report", "doc_type_label", "doc_type_labels", "UNKNOWN_DOC_LABEL"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成单文件离线结果网页")
    parser.add_argument("--out", required=True, help="数据区目录，如 out/dev")
    parser.add_argument("--dest", default=None, help="输出目录（默认 <out>/final）")
    args = parser.parse_args(argv)
    dest = Path(args.dest) if args.dest else None
    result = build_report(Path(args.out), dest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
