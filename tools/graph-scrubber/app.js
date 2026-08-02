const svgNs = "http://www.w3.org/2000/svg";

const state = {
  selectedSentence: 1,
  query: "",
  activeNodeId: null,
  activeEdgeId: null,
};

let graphData = null;
let nodeById = new Map();
let edgeById = new Map();
let positions = new Map();

const typeColors = {
  Book: "#6f52b5",
  Chapter: "#8b6c2b",
  Paragraph: "#7c7165",
  Sentence: "#0f8f7a",
  Quote: "#c7467a",
  Proposition: "#b77a19",
  Character: "#3d7f48",
  CharacterType: "#6d8f38",
  Place: "#3578a8",
  PlaceType: "#4c9aa8",
  Object: "#9f5a9d",
  ObjectType: "#b46d88",
  Event: "#c5622b",
  NounPhrase: "#d94f45",
  Modifier: "#5f6f7d",
  UnresolvedReference: "#8a8178",
};

const lanes = [
  { key: "Structure", x: 115, types: new Set(["Book", "Chapter", "Paragraph"]) },
  { key: "Sentence", x: 315, types: new Set(["Sentence"]) },
  { key: "Quote", x: 500, types: new Set(["Quote"]) },
  { key: "Proposition", x: 690, types: new Set(["Proposition"]) },
  { key: "NounPhrase", x: 890, types: new Set(["NounPhrase"]) },
  { key: "Modifier", x: 1085, types: new Set(["Modifier"]) },
  {
    key: "Entity",
    x: 1270,
    types: new Set(["Character", "Place", "Object", "Event", "UnresolvedReference"]),
  },
  {
    key: "Type",
    x: 1450,
    types: new Set(["CharacterType", "PlaceType", "ObjectType"]),
  },
];

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  bindElements();
  bindEvents();
  await loadData();
  render();
});

function bindElements() {
  for (const id of [
    "totalSentences",
    "totalNodes",
    "totalEdges",
    "prevSentence",
    "nextSentence",
    "sentenceRange",
    "sentenceTicks",
    "sentenceBadge",
    "sentenceLocation",
    "graphSearch",
    "newNodeCount",
    "newEdgeCount",
    "visibleNodeCount",
    "visibleEdgeCount",
    "sentenceText",
    "legend",
    "graphSvg",
    "newHeading",
    "newNodesList",
    "newEdgesList",
    "inspectorBody",
  ]) {
    els[id] = document.getElementById(id);
  }
}

function bindEvents() {
  els.prevSentence.addEventListener("click", () => setSentence(state.selectedSentence - 1));
  els.nextSentence.addEventListener("click", () => setSentence(state.selectedSentence + 1));
  els.sentenceRange.addEventListener("input", (event) => {
    setSentence(Number(event.currentTarget.value));
  });
  els.graphSearch.addEventListener("input", (event) => {
    state.query = event.currentTarget.value.trim().toLowerCase();
    renderGraph();
    renderInspector();
  });
}

async function loadData() {
  const response = await fetch("graph_review.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load graph_review.json: ${response.status}`);
  }
  graphData = await response.json();
  nodeById = new Map(graphData.nodes.map((node) => [node.id, node]));
  edgeById = new Map(graphData.edges.map((edge) => [edge.id, edge]));
  state.selectedSentence = 1;
  els.totalSentences.textContent = String(graphData.metadata.sentence_count);
  els.totalNodes.textContent = String(graphData.metadata.node_count);
  els.totalEdges.textContent = String(graphData.metadata.edge_count);
  els.sentenceRange.min = "1";
  els.sentenceRange.max = String(graphData.metadata.sentence_count);
  els.sentenceRange.value = "1";
  document.documentElement.style.setProperty(
    "--sentence-count",
    String(graphData.metadata.sentence_count),
  );
  computeLayout();
}

function computeLayout() {
  positions = new Map();
  const laneBuckets = new Map(lanes.map((lane) => [lane.key, []]));
  const fallback = laneBuckets.get("Entity");

  for (const node of graphData.nodes) {
    const lane = lanes.find((candidate) => candidate.types.has(node.type));
    (lane ? laneBuckets.get(lane.key) : fallback).push(node);
  }

  let maxHeight = 0;
  for (const lane of lanes) {
    const nodes = laneBuckets.get(lane.key).sort((a, b) => {
      const first = a.first_seen_sentence - b.first_seen_sentence;
      return first || a.type.localeCompare(b.type) || a.label.localeCompare(b.label);
    });
    const rowGap = lane.key === "Sentence" ? 86 : 58;
    const top = 110;
    nodes.forEach((node, index) => {
      const y = top + index * rowGap;
      positions.set(node.id, { x: lane.x, y });
      maxHeight = Math.max(maxHeight, y + 90);
    });
  }

  els.graphSvg.setAttribute("viewBox", `0 0 1600 ${Math.max(700, maxHeight)}`);
  els.graphSvg.style.height = `${Math.max(700, maxHeight)}px`;
}

function render() {
  renderLegend();
  renderTicks();
  renderSentence();
  renderGraph();
  renderLists();
  renderInspector();
}

function renderLegend() {
  replaceChildren(els.legend);
  for (const item of [
    ["New", typeColors.NounPhrase],
    ["Current", typeColors.Sentence],
    ["Seen", typeColors.Place],
    ["Quote", typeColors.Quote],
    ["Proposition", typeColors.Proposition],
    ["Entity", typeColors.Character],
    ["Type", typeColors.CharacterType],
  ]) {
    const row = div("legend-item");
    const dot = div("legend-dot");
    dot.style.background = item[1];
    row.append(dot, textNode(item[0]));
    els.legend.append(row);
  }
}

function renderTicks() {
  replaceChildren(els.sentenceTicks);
  for (const sentence of graphData.sentences) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(sentence.index);
    button.className = sentence.index === state.selectedSentence ? "active" : "";
    button.setAttribute("aria-label", `Sentence ${sentence.index}`);
    button.addEventListener("click", () => setSentence(sentence.index));
    els.sentenceTicks.append(button);
  }
}

function renderSentence() {
  const sentence = getSelectedSentence();
  const summary = getSelectedSummary();
  els.sentenceBadge.textContent = `S${sentence.index} / ${graphData.metadata.sentence_count}`;
  els.sentenceLocation.textContent = `Chapter ${sentence.chapter_num}, paragraph ${sentence.paragraph_num}`;
  els.sentenceText.textContent = sentence.text;
  els.newHeading.textContent = `New in S${sentence.index}`;
  els.newNodeCount.textContent = String(summary.new_node_count);
  els.newEdgeCount.textContent = String(summary.new_edge_count);
  els.sentenceRange.value = String(sentence.index);
  els.prevSentence.disabled = sentence.index <= 1;
  els.nextSentence.disabled = sentence.index >= graphData.metadata.sentence_count;
}

function renderGraph() {
  replaceChildren(els.graphSvg);
  appendMarkers();
  appendLaneLabels();

  const visibleNodes = getVisibleNodes();
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = graphData.edges.filter((edge) => {
    return visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target) && isEdgeVisible(edge);
  });

  els.visibleNodeCount.textContent = String(visibleNodes.length);
  els.visibleEdgeCount.textContent = String(visibleEdges.length);

  const edgeLayer = svgEl("g", { class: "edge-layer" });
  const labelLayer = svgEl("g", { class: "edge-label-layer" });
  const nodeLayer = svgEl("g", { class: "node-layer" });
  els.graphSvg.append(edgeLayer, labelLayer, nodeLayer);

  const matchedNodeIds = getMatchedNodeIds();
  const activeNeighborhood = getActiveNeighborhood(visibleEdges);

  for (const edge of visibleEdges) {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) continue;

    const path = svgEl("path", {
      class: edgeClass(edge, matchedNodeIds, activeNeighborhood),
      d: curvePath(source, target),
      "marker-end": "url(#arrow)",
      "data-edge-id": edge.id,
    });
    path.addEventListener("click", () => {
      state.activeEdgeId = edge.id;
      state.activeNodeId = null;
      renderGraph();
      renderLists();
      renderInspector();
    });
    edgeLayer.append(path);

    if (
      edge.first_seen_sentence === state.selectedSentence ||
      edge.id === state.activeEdgeId ||
      activeNeighborhood.connectedEdgeIds.has(edge.id)
    ) {
      const label = svgEl("text", {
        class: "edge-label",
        x: String((source.x + target.x) / 2),
        y: String((source.y + target.y) / 2 - 8),
        "text-anchor": "middle",
      });
      label.textContent = edge.role ? `${edge.type}:${edge.role}` : edge.type;
      labelLayer.append(label);
    }
  }

  for (const node of visibleNodes) {
    const point = positions.get(node.id);
    if (!point) continue;
    const group = svgEl("g", {
      class: nodeClass(node, matchedNodeIds, activeNeighborhood),
      transform: `translate(${point.x} ${point.y})`,
      "data-node-id": node.id,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.type}: ${node.label}`,
    });
    group.addEventListener("click", () => {
      state.activeNodeId = state.activeNodeId === node.id ? null : node.id;
      state.activeEdgeId = null;
      renderGraph();
      renderLists();
      renderInspector();
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        state.activeNodeId = state.activeNodeId === node.id ? null : node.id;
        state.activeEdgeId = null;
        renderGraph();
        renderLists();
        renderInspector();
      }
    });

    const shape = svgEl(node.type === "Proposition" ? "rect" : "circle", nodeShapeAttrs(node));
    const halo = svgEl("circle", { class: "node-halo", r: "25" });
    const label = svgEl("text", {
      class: "node-label",
      x: "0",
      y: "35",
      "text-anchor": "middle",
    });
    label.textContent = truncate(node.label, 20);
    const subtitle = svgEl("text", {
      class: "node-subtitle",
      x: "0",
      y: "49",
      "text-anchor": "middle",
    });
    subtitle.textContent = node.type;
    group.append(halo, shape, label, subtitle);
    nodeLayer.append(group);
  }
}

function appendMarkers() {
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "arrow",
    viewBox: "0 0 10 10",
    refX: "8",
    refY: "5",
    markerWidth: "6",
    markerHeight: "6",
    orient: "auto-start-reverse",
  });
  marker.append(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "currentColor" }));
  defs.append(marker);
  els.graphSvg.append(defs);
}

function appendLaneLabels() {
  const layer = svgEl("g", { class: "lane-labels" });
  for (const lane of lanes) {
    const label = svgEl("text", {
      x: String(lane.x),
      y: "62",
      "text-anchor": "middle",
      class: "node-subtitle",
    });
    label.textContent = lane.key;
    layer.append(label);
  }
  els.graphSvg.append(layer);
}

function renderLists() {
  const summary = getSelectedSummary();
  const newNodes = summary.new_node_ids.map((id) => nodeById.get(id)).filter(Boolean);
  const newEdges = summary.new_edge_ids.map((id) => edgeById.get(id)).filter(Boolean);

  replaceChildren(els.newNodesList);
  if (newNodes.length === 0) {
    els.newNodesList.append(emptyState("No new nodes."));
  } else {
    for (const node of newNodes) {
      const item = pill(`${node.label}`, `${node.type} - first seen S${node.first_seen_sentence}`);
      item.classList.toggle("active", state.activeNodeId === node.id);
      item.addEventListener("click", () => {
        state.activeNodeId = node.id;
        state.activeEdgeId = null;
        renderGraph();
        renderLists();
        renderInspector();
      });
      els.newNodesList.append(item);
    }
  }

  replaceChildren(els.newEdgesList);
  if (newEdges.length === 0) {
    els.newEdgesList.append(emptyState("No new edges."));
  } else {
    for (const edge of newEdges) {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      const title = edge.role ? `${edge.type}:${edge.role}` : edge.type;
      const meta = `${source?.label || edge.source} -> ${target?.label || edge.target}`;
      const item = pill(title, meta);
      item.classList.toggle("active", state.activeEdgeId === edge.id);
      item.addEventListener("click", () => {
        state.activeEdgeId = edge.id;
        state.activeNodeId = null;
        renderGraph();
        renderLists();
        renderInspector();
      });
      els.newEdgesList.append(item);
    }
  }
}

function renderInspector() {
  replaceChildren(els.inspectorBody);
  if (state.activeNodeId && nodeById.has(state.activeNodeId)) {
    const node = nodeById.get(state.activeNodeId);
    els.inspectorBody.append(kvList({
      Type: node.type,
      Label: node.label,
      "First seen": `S${node.first_seen_sentence}`,
      Seen: node.sentence_indices.map((value) => `S${value}`).join(", "),
      ID: node.id,
      Properties: formatProperties(node.properties),
    }));
    return;
  }

  if (state.activeEdgeId && edgeById.has(state.activeEdgeId)) {
    const edge = edgeById.get(state.activeEdgeId);
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    els.inspectorBody.append(kvList({
      Type: edge.role ? `${edge.type}:${edge.role}` : edge.type,
      Source: source ? source.label : edge.source,
      Target: target ? target.label : edge.target,
      "First seen": `S${edge.first_seen_sentence}`,
      Seen: edge.sentence_indices.map((value) => `S${value}`).join(", "),
      ID: edge.id,
      Properties: formatProperties(edge.properties),
    }));
    return;
  }

  const sentence = getSelectedSentence();
  els.inspectorBody.append(kvList({
    Selected: `S${sentence.index}`,
    Passage: sentence.id,
    Chapter: `${sentence.chapter_num}`,
    Paragraph: `${sentence.paragraph_num}`,
  }));
}

function setSentence(value) {
  const next = Math.max(1, Math.min(graphData.metadata.sentence_count, value));
  if (next === state.selectedSentence) return;
  state.selectedSentence = next;
  state.activeNodeId = null;
  state.activeEdgeId = null;
  renderTicks();
  renderSentence();
  renderGraph();
  renderLists();
  renderInspector();
}

function getSelectedSentence() {
  return graphData.sentences.find((sentence) => sentence.index === state.selectedSentence);
}

function getSelectedSummary() {
  return graphData.sentence_summaries.find((summary) => summary.index === state.selectedSentence);
}

function getVisibleNodes() {
  return graphData.nodes.filter((node) => {
    return node.first_seen_sentence <= state.selectedSentence;
  });
}

function isEdgeVisible(edge) {
  return edge.first_seen_sentence <= state.selectedSentence;
}

function getMatchedNodeIds() {
  if (!state.query) return new Set();
  const matches = new Set();
  for (const node of graphData.nodes) {
    const haystack = `${node.label} ${node.type} ${node.id} ${formatProperties(node.properties)}`.toLowerCase();
    if (haystack.includes(state.query)) {
      matches.add(node.id);
    }
  }
  return matches;
}

function getActiveNeighborhood(visibleEdges) {
  const connectedNodeIds = new Set();
  const connectedEdgeIds = new Set();
  if (!state.activeNodeId) {
    return { connectedNodeIds, connectedEdgeIds };
  }

  for (const edge of visibleEdges) {
    if (edge.source !== state.activeNodeId && edge.target !== state.activeNodeId) {
      continue;
    }
    connectedEdgeIds.add(edge.id);
    connectedNodeIds.add(edge.source);
    connectedNodeIds.add(edge.target);
  }

  return { connectedNodeIds, connectedEdgeIds };
}

function nodeClass(node, matchedNodeIds, activeNeighborhood) {
  const classes = ["node-group"];
  if (node.sentence_indices.includes(state.selectedSentence)) classes.push("current");
  if (node.first_seen_sentence === state.selectedSentence) classes.push("new");
  if (state.activeNodeId === node.id) classes.push("active");
  if (state.activeNodeId && activeNeighborhood.connectedNodeIds.has(node.id)) {
    classes.push("connected");
  }
  if (state.activeNodeId && !activeNeighborhood.connectedNodeIds.has(node.id)) {
    classes.push("dimmed");
  }
  if (state.activeEdgeId) {
    const edge = edgeById.get(state.activeEdgeId);
    if (edge && node.id !== edge.source && node.id !== edge.target) {
      classes.push("dimmed");
    }
    if (edge && (node.id === edge.source || node.id === edge.target)) {
      classes.push("connected");
    }
  }
  if (state.query && !matchedNodeIds.has(node.id)) classes.push("dimmed");
  return classes.join(" ");
}

function edgeClass(edge, matchedNodeIds, activeNeighborhood) {
  const classes = ["graph-edge"];
  if (edge.sentence_indices.includes(state.selectedSentence)) classes.push("current");
  if (edge.first_seen_sentence === state.selectedSentence) classes.push("new");
  if (state.activeEdgeId === edge.id) classes.push("new");
  if (state.activeNodeId && activeNeighborhood.connectedEdgeIds.has(edge.id)) {
    classes.push("connected");
  }
  if (state.activeNodeId && !activeNeighborhood.connectedEdgeIds.has(edge.id)) {
    classes.push("dimmed");
  }
  if (state.activeEdgeId && state.activeEdgeId !== edge.id) {
    classes.push("dimmed");
  }
  if (
    state.query &&
    !matchedNodeIds.has(edge.source) &&
    !matchedNodeIds.has(edge.target) &&
    !`${edge.type} ${edge.role || ""}`.toLowerCase().includes(state.query)
  ) {
    classes.push("dimmed");
  }
  return classes.join(" ");
}

function nodeShapeAttrs(node) {
  const color = typeColors[node.type] || "#8a8178";
  if (node.type === "Proposition") {
    return {
      class: "node-shape",
      x: "-21",
      y: "-17",
      width: "42",
      height: "34",
      rx: "7",
      fill: color,
    };
  }
  const radius = node.type === "Sentence" ? 19 : node.type === "Book" ? 21 : 16;
  return {
    class: "node-shape",
    r: String(radius),
    fill: color,
  };
}

function curvePath(source, target) {
  const dx = Math.max(80, Math.abs(target.x - source.x) * 0.48);
  return `M ${source.x} ${source.y} C ${source.x + dx} ${source.y}, ${target.x - dx} ${target.y}, ${target.x} ${target.y}`;
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(svgNs, name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

function div(className) {
  const el = document.createElement("div");
  el.className = className;
  return el;
}

function textNode(value) {
  return document.createTextNode(value);
}

function replaceChildren(el) {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

function truncate(value, limit) {
  if (!value) return "";
  return value.length > limit ? `${value.slice(0, limit - 1)}...` : value;
}

function emptyState(message) {
  const el = div("empty-state");
  el.textContent = message;
  return el;
}

function pill(title, meta) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "pill";
  const titleEl = div("pill-title");
  titleEl.textContent = title;
  const metaEl = div("pill-meta");
  metaEl.textContent = meta;
  button.append(titleEl, metaEl);
  return button;
}

function kvList(rows) {
  const dl = document.createElement("dl");
  for (const [key, value] of Object.entries(rows)) {
    if (value === undefined || value === null || value === "") continue;
    const row = div("kv");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = String(value);
    row.append(dt, dd);
    dl.append(row);
  }
  return dl;
}

function formatProperties(properties) {
  if (!properties || Object.keys(properties).length === 0) return "";
  return JSON.stringify(properties);
}
