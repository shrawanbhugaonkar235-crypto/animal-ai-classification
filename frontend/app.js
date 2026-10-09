const API_BASE = (window.ANIMAL_AI_API || "").replace(/\/$/, "");
const api = (path) => API_BASE + path;
const $ = (id) => document.getElementById(id);

let selected = null;
let detections = [];

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

function showView(view) {
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === view);
  });
  document.querySelectorAll("nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  $("nav").classList.remove("open");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("menu").addEventListener("click", () => $("nav").classList.toggle("open"));
$("browse").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => {
  if ($("file").files[0]) setFile($("file").files[0]);
});
$("remove").addEventListener("click", reset);
$("newPhoto").addEventListener("click", reset);
$("analyze").addEventListener("click", analyze);

["search", "category", "confidence"].forEach((id) => {
  $(id).addEventListener("input", renderList);
});

$("drop").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("drop").classList.add("drag");
});

$("drop").addEventListener("dragleave", () => $("drop").classList.remove("drag"));

$("drop").addEventListener("drop", (event) => {
  event.preventDefault();
  $("drop").classList.remove("drag");
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

$("drop").addEventListener("click", (event) => {
  if (
    event.target === $("drop") ||
    event.target.tagName === "H3" ||
    event.target.tagName === "P" ||
    event.target.classList.contains("upload-icon")
  ) {
    $("file").click();
  }
});

function setFile(file) {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    return showError("Please choose JPG, JPEG, PNG or WEBP.");
  }

  selected = file;
  $("filename").textContent = file.name + " · " + (file.size / 1024 / 1024).toFixed(2) + " MB";
  $("preview").src = URL.createObjectURL(file);
  $("drop").classList.add("hidden");
  $("previewBox").classList.remove("hidden");
  $("results").classList.add("hidden");
  hideError();
}

function reset() {
  selected = null;
  $("file").value = "";
  $("preview").removeAttribute("src");
  $("drop").classList.remove("hidden");
  $("previewBox").classList.add("hidden");
  $("results").classList.add("hidden");
  hideError();
}

function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("hidden");
}

function hideError() {
  $("error").textContent = "";
  $("error").classList.add("hidden");
}

async function compressImage(file) {
  const bitmap = await createImageBitmap(file);
  const maxSide = 1600;
  const scale = Math.min(1, maxSide / bitmap.width, maxSide / bitmap.height);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  let quality = 0.82;
  let blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));

  while (blob && blob.size > 4 * 1024 * 1024 && quality > 0.45) {
    quality -= 0.08;
    blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  }

  if (!blob || blob.size > 4 * 1024 * 1024) {
    throw new Error("Please choose a smaller photo.");
  }

  return new File([blob], "upload.jpg", { type: "image/jpeg" });
}

async function analyze() {
  if (!selected) return showError("Select a photo first.");

  $("loading").classList.remove("hidden");
  $("analyze").disabled = true;
  hideError();

  try {
    const file = await compressImage(selected);
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(api("/api/predict"), {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    detections = data.detections || [];
    $("results").classList.remove("hidden");
    $("total").textContent = data.total_detections;
    $("land").textContent = data.summary.land_animals;
    $("sea").textContent = data.summary.sea_animals;
    $("birds").textContent = data.summary.birds;
    $("avg").textContent = data.summary.average_confidence + "%";
    $("ptime").textContent = data.processing_time_ms + " ms";
    $("model").textContent = data.model_name;
    $("output").src = data.annotated_image;
    renderList();
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message || "Could not analyze the image.");
  } finally {
    $("loading").classList.add("hidden");
    $("analyze").disabled = false;
  }
}

function renderList() {
  const query = $("search").value.trim().toLowerCase();
  const category = $("category").value;
  const minimum = Number($("confidence").value);

  const rows = detections.filter((item) => {
    return (
      item.name.toLowerCase().includes(query) &&
      (category === "All" || item.category === category) &&
      item.confidence_percent >= minimum
    );
  });

  $("list").innerHTML = rows.length
    ? rows.map((item) => {
        const box = item.bbox;
        return `
          <div class="item">
            <div class="itemtop">
              <div>
                <b>${escapeHtml(item.name)}</b>
                <small>${escapeHtml(item.category)} · ${escapeHtml(item.status)}</small>
              </div>
              <b>${item.confidence_percent}%</b>
            </div>
            <div class="prog"><span style="width:${Math.min(100, item.confidence_percent)}%"></span></div>
            <div class="bbox">Box: [${box.x1}, ${box.y1}] → [${box.x2}, ${box.y2}]</div>
          </div>
        `;
      }).join("")
    : '<div class="panel" style="box-shadow:none;text-align:center">No detections match the filters.</div>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[char]));
}

async function load() {
  try {
    const healthResponse = await fetch(api("/api/health"));
    const health = await healthResponse.json();

    $("health").textContent = health.model_loaded
      ? "Model ready · " + health.model_name
      : "Model unavailable";
    $("health").className = "status " + (health.model_loaded ? "ok" : "bad");

    const classesResponse = await fetch(api("/api/classes"));
    const data = await classesResponse.json();

    const actual = new Set((data.actual_model_classes || []).map((name) => name.toLowerCase().trim()));

    $("coverage").innerHTML = data.model_loaded
      ? "<b>Loaded model:</b> " + escapeHtml(data.model_name) +
        " · <b>" + actual.size + "</b> actual classes. Classes not in the model are marked as training-required."
      : "<b>AI model unavailable.</b> Add trained weights or configure a compatible model.";

    const groups = {
      landClasses: [],
      seaClasses: [],
      birdClasses: []
    };

    for (const item of (data.classes || [])) {
      const target =
        item.category === "Land Animals" ? "landClasses" :
        item.category === "Sea Animals" ? "seaClasses" : "birdClasses";
      groups[target].push(item);
    }

    for (const [id, items] of Object.entries(groups)) {
      $(id).innerHTML = items.map((item) => {
        const supported = actual.has(item.name.toLowerCase().trim());
        return `
          <div class="chip ${supported ? "supported" : ""}">
            <b>${escapeHtml(item.name)}</b>
            <small>${supported ? "✓ Supported" : "○ Training required"}</small>
          </div>
        `;
      }).join("");
    }
  } catch (error) {
    $("health").textContent = "API unavailable";
    $("health").className = "status bad";
  }
}

load();
