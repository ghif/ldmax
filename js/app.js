/**
 * LDMAX Web Frontend Application Script
 * Connects to FastAPI Backend on Google Cloud Run or Local Host
 */

// Default constants matching scripts/demo.py
const CIFAR10_CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
];

const FASHION_CLASSES = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

const CELEBA_ATTRIBUTES = [
  "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
  "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
  "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
  "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
  "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
  "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
  "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
  "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
];

// App State
const state = {
  apiBaseUrl: localStorage.getItem("ldmax_api_url") || (
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:8000"
      : "https://ldmax-backend-19423981787.asia-southeast2.run.app"
  ),
  theme: localStorage.getItem("ldmax_theme") || "dark",
  activeTab: "cifar10",
  backendStatus: "connecting",
  cifar10: {
    weights: [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    samples: 4,
    steps: 20,
    cfg: 1.5,
    seed: 0,
    generatedImages: [],
    lastCaption: "airplane (1.00)",
    lastLatency: 0,
  },
  fashion_mnist: {
    weights: [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
    samples: 4,
    steps: 20,
    cfg: 1.5,
    seed: 0,
    generatedImages: [],
    lastCaption: "Sneaker (1.00)",
    lastLatency: 0,
  },
  celeba: {
    selectedAttributes: new Set(["Smiling", "Young"]),
    samples: 2,
    steps: 20,
    cfg: 4.0,
    seed: 42,
    generatedImages: [],
    lastCaption: "Smiling, Young",
    lastLatency: 0,
  },
  lightbox: {
    images: [],
    currentIndex: 0,
    scale: 1.0,
    rotation: 0,
    flipH: 1,
    opacity: 1.0,
    isInverted: false,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
    isCrisp: true,
    isPlaying: false,
    playInterval: null,
    currentTab: "cifar10",
  }
};

let elements = {};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initDOMReferences();
  initTheme();
  initTabs();
  initCIFAR10Controls();
  initFashionControls();
  initCelebAControls();
  initSettingsModal();
  initLightbox();
  checkBackendHealth();
});

function initDOMReferences() {
  elements = {
    statusPill: document.getElementById("backend-status-pill"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    settingsBtn: document.getElementById("settings-btn"),
    settingsModal: document.getElementById("settings-modal"),
    backendUrlInput: document.getElementById("backend-url-input"),
    saveSettingsBtn: document.getElementById("save-settings-btn"),
    testConnBtn: document.getElementById("test-connection-btn"),
    closeSettingsModal: document.getElementById("close-settings-modal"),
    modalStatusPreview: document.getElementById("modal-status-preview"),

    // Lightbox & Detail View Elements
    lightboxModal: document.getElementById("lightbox-modal"),
    lightboxContainer: document.getElementById("lightbox-container"),
    lightboxBackdrop: document.getElementById("lightbox-backdrop"),
    lightboxImg: document.getElementById("lightbox-img"),
    lightboxWrapper: document.getElementById("lightbox-image-wrapper"),
    lightboxViewport: document.getElementById("lightbox-viewport"),
    lightboxDownloadBtn: document.getElementById("lightbox-download-btn"),
    closeLightbox: document.getElementById("close-lightbox"),
    lightboxPrevBtn: document.getElementById("lightbox-prev-btn"),
    lightboxNextBtn: document.getElementById("lightbox-next-btn"),
    lightboxPlayBtn: document.getElementById("lightbox-play-btn"),
    lightboxCounter: document.getElementById("lightbox-counter"),
    lightboxZoomInBtn: document.getElementById("lightbox-zoom-in-btn"),
    lightboxZoomOutBtn: document.getElementById("lightbox-zoom-out-btn"),
    lightboxZoomResetBtn: document.getElementById("lightbox-zoom-reset-btn"),
    lightboxZoomLevel: document.getElementById("lightbox-zoom-level"),
    lightboxPixelToggleBtn: document.getElementById("lightbox-pixel-toggle-btn"),
    lightboxRotateBtn: document.getElementById("lightbox-rotate-btn"),
    lightboxFliphBtn: document.getElementById("lightbox-fliph-btn"),
    lightboxInvertBtn: document.getElementById("lightbox-invert-btn"),
    lightboxFadeSlider: document.getElementById("lightbox-fade-slider"),
    lightboxCopyBtn: document.getElementById("lightbox-copy-btn"),
    lightboxInfoBtn: document.getElementById("lightbox-info-btn"),
    lightboxFullscreenBtn: document.getElementById("lightbox-fullscreen-btn"),
    lightboxInfoDrawer: document.getElementById("lightbox-info-drawer"),
    infoDrawerContent: document.getElementById("info-drawer-content"),
    closeInfoDrawer: document.getElementById("close-info-drawer"),
  };
}

// ---------------------------------------------------------------------------
// Theme & Navigation
// ---------------------------------------------------------------------------

function initTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
  if (elements.themeToggleBtn) {
    elements.themeToggleBtn.textContent = state.theme === "dark" ? "🌙" : "☀️";
    elements.themeToggleBtn.addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", state.theme);
      localStorage.setItem("ldmax_theme", state.theme);
      elements.themeToggleBtn.textContent = state.theme === "dark" ? "🌙" : "☀️";
    });
  }
}

function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabKey = btn.dataset.tab;
      state.activeTab = tabKey;

      tabButtons.forEach(b => {
        const isActive = b.dataset.tab === tabKey;
        b.classList.toggle("active", isActive);
        b.setAttribute("aria-selected", isActive);
      });

      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `panel-${tabKey}`);
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Backend Health Check & Settings
// ---------------------------------------------------------------------------

async function checkBackendHealth() {
  setBackendStatus("connecting", "Connecting...");
  try {
    const res = await fetch(`${state.apiBaseUrl}/api/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      const platform = data.default_platform ? data.default_platform.toUpperCase() : "OK";
      setBackendStatus("connected", `Connected (${platform})`);
    } else {
      setBackendStatus("disconnected", "Backend Error");
    }
  } catch (err) {
    setBackendStatus("disconnected", "Offline / Check URL");
  }
}

function setBackendStatus(status, text) {
  state.backendStatus = status;
  if (!elements.statusPill) return;
  elements.statusPill.className = `status-pill status-${status}`;
  const textElem = elements.statusPill.querySelector(".status-text");
  if (textElem) textElem.textContent = text;
}

function initSettingsModal() {
  if (!elements.settingsBtn) return;
  elements.backendUrlInput.value = state.apiBaseUrl;

  elements.settingsBtn.addEventListener("click", () => {
    elements.backendUrlInput.value = state.apiBaseUrl;
    elements.modalStatusPreview.innerHTML = "";
    elements.settingsModal.classList.remove("hidden");
  });

  const closeModal = () => elements.settingsModal.classList.add("hidden");
  elements.closeSettingsModal.addEventListener("click", closeModal);
  elements.settingsModal.querySelector(".modal-backdrop").addEventListener("click", closeModal);

  elements.testConnBtn.addEventListener("click", async () => {
    const testUrl = elements.backendUrlInput.value.trim().replace(/\/+$/, "");
    elements.modalStatusPreview.innerHTML = '<span style="color: var(--warning)">Testing connection...</span>';
    try {
      const res = await fetch(`${testUrl}/api/health`);
      if (res.ok) {
        const data = await res.json();
        elements.modalStatusPreview.innerHTML = `<span style="color: var(--success)">✅ Successfully reached backend! Platform: ${data.default_platform}</span>`;
      } else {
        elements.modalStatusPreview.innerHTML = `<span style="color: var(--danger)">❌ Backend returned status ${res.status}</span>`;
      }
    } catch (err) {
      elements.modalStatusPreview.innerHTML = `<span style="color: var(--danger)">❌ Connection failed: ${err.message}</span>`;
    }
  });

  elements.saveSettingsBtn.addEventListener("click", () => {
    state.apiBaseUrl = elements.backendUrlInput.value.trim().replace(/\/+$/, "");
    localStorage.setItem("ldmax_api_url", state.apiBaseUrl);
    closeModal();
    checkBackendHealth();
  });
}

// ---------------------------------------------------------------------------
// CIFAR-10 Tab Controls
// ---------------------------------------------------------------------------

function initCIFAR10Controls() {
  const container = document.getElementById("cifar10-sliders-container");
  if (!container) return;
  container.innerHTML = "";

  CIFAR10_CLASSES.forEach((name, idx) => {
    const item = document.createElement("div");
    item.className = "slider-item";
    item.innerHTML = `
      <div class="slider-label-row">
        <span>${idx}: ${name}</span>
        <span class="val" id="cifar10-val-${idx}">${state.cifar10.weights[idx].toFixed(2)}</span>
      </div>
      <input type="range" id="cifar10-slider-${idx}" min="0" max="1" step="0.05" value="${state.cifar10.weights[idx]}">
    `;
    container.appendChild(item);

    const slider = item.querySelector("input");
    slider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      state.cifar10.weights[idx] = val;
      document.getElementById(`cifar10-val-${idx}`).textContent = val.toFixed(2);
      clearActivePresets("cifar10-presets");
    });
  });

  // Presets
  const presetsContainer = document.getElementById("cifar10-presets");
  presetsContainer.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const classIdx = parseInt(chip.dataset.class);
      state.cifar10.weights = state.cifar10.weights.map((_, i) => i === classIdx ? 1.0 : 0.0);
      CIFAR10_CLASSES.forEach((_, i) => {
        document.getElementById(`cifar10-slider-${i}`).value = state.cifar10.weights[i];
        document.getElementById(`cifar10-val-${i}`).textContent = state.cifar10.weights[i].toFixed(2);
      });
      presetsContainer.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
    });
  });

  // Reset Button
  document.getElementById("cifar10-reset-sliders").addEventListener("click", () => {
    state.cifar10.weights = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    CIFAR10_CLASSES.forEach((_, i) => {
      document.getElementById(`cifar10-slider-${i}`).value = state.cifar10.weights[i];
      document.getElementById(`cifar10-val-${i}`).textContent = state.cifar10.weights[i].toFixed(2);
    });
    presetsContainer.querySelectorAll(".chip").forEach((c, i) => c.classList.toggle("active", i === 0));
  });

  // Sliders binding
  bindSlider("cifar10-samples", "cifar10-samples-val", (v) => state.cifar10.samples = parseInt(v));
  bindSlider("cifar10-steps", "cifar10-steps-val", (v) => state.cifar10.steps = parseInt(v));
  bindSlider("cifar10-cfg", "cifar10-cfg-val", (v) => state.cifar10.cfg = parseFloat(v));

  // Seed Randomize
  document.getElementById("cifar10-seed-randomize").addEventListener("click", () => {
    const newSeed = Math.floor(Math.random() * 100000);
    document.getElementById("cifar10-seed").value = newSeed;
    state.cifar10.seed = newSeed;
  });

  // Generate Button
  document.getElementById("cifar10-generate-btn").addEventListener("click", async () => {
    state.cifar10.seed = parseInt(document.getElementById("cifar10-seed").value) || 0;
    await triggerGeneration({
      tabKey: "cifar10",
      btnId: "cifar10-generate-btn",
      url: `${state.apiBaseUrl}/api/generate/cifar10`,
      body: {
        class_weights: state.cifar10.weights,
        num_samples: state.cifar10.samples,
        inference_steps: state.cifar10.steps,
        cfg_scale: state.cifar10.cfg,
        seed: state.cifar10.seed,
      },
      galleryId: "cifar10-gallery",
      captionId: "cifar10-caption",
      latencyId: "cifar10-latency",
    });
  });
}

// ---------------------------------------------------------------------------
// Fashion-MNIST Tab Controls
// ---------------------------------------------------------------------------

function initFashionControls() {
  const container = document.getElementById("fashion-sliders-container");
  if (!container) return;
  container.innerHTML = "";

  FASHION_CLASSES.forEach((name, idx) => {
    const item = document.createElement("div");
    item.className = "slider-item";
    item.innerHTML = `
      <div class="slider-label-row">
        <span>${idx}: ${name}</span>
        <span class="val" id="fashion-val-${idx}">${state.fashion_mnist.weights[idx].toFixed(2)}</span>
      </div>
      <input type="range" id="fashion-slider-${idx}" min="0" max="1" step="0.05" value="${state.fashion_mnist.weights[idx]}">
    `;
    container.appendChild(item);

    const slider = item.querySelector("input");
    slider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      state.fashion_mnist.weights[idx] = val;
      document.getElementById(`fashion-val-${idx}`).textContent = val.toFixed(2);
      clearActivePresets("fashion-presets");
    });
  });

  // Presets
  const presetsContainer = document.getElementById("fashion-presets");
  presetsContainer.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const classIdx = parseInt(chip.dataset.class);
      state.fashion_mnist.weights = state.fashion_mnist.weights.map((_, i) => i === classIdx ? 1.0 : 0.0);
      FASHION_CLASSES.forEach((_, i) => {
        document.getElementById(`fashion-slider-${i}`).value = state.fashion_mnist.weights[i];
        document.getElementById(`fashion-val-${i}`).textContent = state.fashion_mnist.weights[i].toFixed(2);
      });
      presetsContainer.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
    });
  });

  // Reset Button
  document.getElementById("fashion-reset-sliders").addEventListener("click", () => {
    state.fashion_mnist.weights = [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0];
    FASHION_CLASSES.forEach((_, i) => {
      document.getElementById(`fashion-slider-${i}`).value = state.fashion_mnist.weights[i];
      document.getElementById(`fashion-val-${i}`).textContent = state.fashion_mnist.weights[i].toFixed(2);
    });
    presetsContainer.querySelectorAll(".chip").forEach((c, i) => c.classList.toggle("active", i === 7));
  });

  // Sliders binding
  bindSlider("fashion-samples", "fashion-samples-val", (v) => state.fashion_mnist.samples = parseInt(v));
  bindSlider("fashion-steps", "fashion-steps-val", (v) => state.fashion_mnist.steps = parseInt(v));
  bindSlider("fashion-cfg", "fashion-cfg-val", (v) => state.fashion_mnist.cfg = parseFloat(v));

  // Seed Randomize
  document.getElementById("fashion-seed-randomize").addEventListener("click", () => {
    const newSeed = Math.floor(Math.random() * 100000);
    document.getElementById("fashion-seed").value = newSeed;
    state.fashion_mnist.seed = newSeed;
  });

  // Generate Button
  document.getElementById("fashion-generate-btn").addEventListener("click", async () => {
    state.fashion_mnist.seed = parseInt(document.getElementById("fashion-seed").value) || 0;
    await triggerGeneration({
      tabKey: "fashion_mnist",
      btnId: "fashion-generate-btn",
      url: `${state.apiBaseUrl}/api/generate/fashion_mnist`,
      body: {
        class_weights: state.fashion_mnist.weights,
        num_samples: state.fashion_mnist.samples,
        inference_steps: state.fashion_mnist.steps,
        cfg_scale: state.fashion_mnist.cfg,
        seed: state.fashion_mnist.seed,
      },
      galleryId: "fashion-gallery",
      captionId: "fashion-caption",
      latencyId: "fashion-latency",
    });
  });
}

// ---------------------------------------------------------------------------
// CelebA Tab Controls
// ---------------------------------------------------------------------------

function initCelebAControls() {
  const container = document.getElementById("celeba-attributes-container");
  if (!container) return;
  container.innerHTML = "";

  CELEBA_ATTRIBUTES.forEach(attrName => {
    const chip = document.createElement("button");
    chip.className = `attr-chip ${state.celeba.selectedAttributes.has(attrName) ? "selected" : ""}`;
    chip.textContent = attrName.replace(/_/g, " ");
    chip.dataset.attr = attrName;

    chip.addEventListener("click", () => {
      if (state.celeba.selectedAttributes.has(attrName)) {
        state.celeba.selectedAttributes.delete(attrName);
        chip.classList.remove("selected");
      } else {
        state.celeba.selectedAttributes.add(attrName);
        chip.classList.add("selected");
      }
      clearActivePresets("celeba-presets");
    });
    container.appendChild(chip);
  });

  // Attribute Search Filter
  document.getElementById("celeba-attr-search").addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    container.querySelectorAll(".attr-chip").forEach(chip => {
      const match = chip.dataset.attr.toLowerCase().includes(query);
      chip.style.display = match ? "inline-block" : "none";
    });
  });

  // Clear Attributes
  document.getElementById("celeba-clear-attrs").addEventListener("click", () => {
    state.celeba.selectedAttributes.clear();
    container.querySelectorAll(".attr-chip").forEach(chip => chip.classList.remove("selected"));
    clearActivePresets("celeba-presets");
  });

  // Presets
  const presetsContainer = document.getElementById("celeba-presets");
  presetsContainer.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const rawAttrs = chip.dataset.attrs;
      const attrList = rawAttrs ? rawAttrs.split(",") : [];
      state.celeba.selectedAttributes = new Set(attrList);

      container.querySelectorAll(".attr-chip").forEach(c => {
        c.classList.toggle("selected", state.celeba.selectedAttributes.has(c.dataset.attr));
      });

      presetsContainer.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
    });
  });

  // Sliders binding
  bindSlider("celeba-samples", "celeba-samples-val", (v) => state.celeba.samples = parseInt(v));
  bindSlider("celeba-steps", "celeba-steps-val", (v) => state.celeba.steps = parseInt(v));
  bindSlider("celeba-cfg", "celeba-cfg-val", (v) => state.celeba.cfg = parseFloat(v));

  // Seed Randomize
  document.getElementById("celeba-seed-randomize").addEventListener("click", () => {
    const newSeed = Math.floor(Math.random() * 100000);
    document.getElementById("celeba-seed").value = newSeed;
    state.celeba.seed = newSeed;
  });

  // Generate Button
  document.getElementById("celeba-generate-btn").addEventListener("click", async () => {
    state.celeba.seed = parseInt(document.getElementById("celeba-seed").value) || 0;
    await triggerGeneration({
      tabKey: "celeba",
      btnId: "celeba-generate-btn",
      url: `${state.apiBaseUrl}/api/generate/celeba`,
      body: {
        selected_attributes: Array.from(state.celeba.selectedAttributes),
        num_samples: state.celeba.samples,
        inference_steps: state.celeba.steps,
        cfg_scale: state.celeba.cfg,
        seed: state.celeba.seed,
      },
      galleryId: "celeba-gallery",
      captionId: "celeba-caption",
      latencyId: "celeba-latency",
    });
  });
}

// ---------------------------------------------------------------------------
// Generation Request & Gallery Rendering
// ---------------------------------------------------------------------------

async function triggerGeneration({ tabKey, btnId, url, body, galleryId, captionId, latencyId }) {
  const btn = document.getElementById(btnId);
  const spinner = btn.querySelector(".btn-spinner");
  const btnText = btn.querySelector(".btn-text");
  const gallery = document.getElementById(galleryId);
  const caption = document.getElementById(captionId);
  const latencyBadge = document.getElementById(latencyId);

  // Set loading state
  btn.disabled = true;
  spinner.classList.remove("hidden");
  const originalText = btnText.textContent;
  btnText.textContent = "Denoising in JAX...";

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errData.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    state[tabKey].generatedImages = data.images;
    state[tabKey].lastCaption = data.caption;
    state[tabKey].lastLatency = data.time_taken_sec;

    // Render Images
    gallery.innerHTML = "";
    data.images.forEach((b64Img, index) => {
      const item = document.createElement("div");
      item.className = "gallery-item";
      item.title = "Click to View Details, Zoom & Fade";
      item.innerHTML = `
        <img src="${b64Img}" alt="Generated sample ${index + 1}" loading="lazy">
        <div class="item-overlay">
          <button class="overlay-btn zoom-btn" title="View Fullsize & Zoom">🔍 Details</button>
          <a class="overlay-btn download-btn" href="${b64Img}" download="ldmax_${tabKey}_sample_${index + 1}.png" title="Download">⬇️ Save</a>
        </div>
      `;

      item.querySelector(".zoom-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        openLightbox(data.images, index, tabKey);
      });
      item.addEventListener("click", () => openLightbox(data.images, index, tabKey));

      gallery.appendChild(item);
    });

    // Update Caption & Latency
    caption.textContent = data.caption;
    latencyBadge.textContent = `⚡ ${data.time_taken_sec.toFixed(2)}s`;
    latencyBadge.classList.remove("hidden");
  } catch (err) {
    alert(`Generation Error: ${err.message}\n\nPlease verify backend is running at ${state.apiBaseUrl}`);
    console.error("Generation failed:", err);
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = originalText;
  }
}

// ---------------------------------------------------------------------------
// Interactive Lightbox, Zoom, Fade & Image Detail Engine
// ---------------------------------------------------------------------------

function initLightbox() {
  if (!elements.lightboxModal) return;

  const closeModal = () => {
    stopSlideshow();
    elements.lightboxModal.classList.add("hidden");
    elements.lightboxInfoDrawer.classList.add("hidden");
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
    resetTransforms();
  };

  elements.closeLightbox.addEventListener("click", closeModal);
  elements.lightboxBackdrop.addEventListener("click", closeModal);

  // Zoom Buttons
  elements.lightboxZoomInBtn.addEventListener("click", () => zoomStep(1.3));
  elements.lightboxZoomOutBtn.addEventListener("click", () => zoomStep(1 / 1.3));
  elements.lightboxZoomResetBtn.addEventListener("click", resetTransforms);

  // Pixelated / Smooth Toggle
  elements.lightboxPixelToggleBtn.addEventListener("click", () => {
    state.lightbox.isCrisp = !state.lightbox.isCrisp;
    updatePixelMode();
  });

  // Rotation & Flip
  elements.lightboxRotateBtn.addEventListener("click", () => {
    state.lightbox.rotation = (state.lightbox.rotation + 90) % 360;
    applyTransform();
  });

  elements.lightboxFliphBtn.addEventListener("click", () => {
    state.lightbox.flipH = state.lightbox.flipH === 1 ? -1 : 1;
    applyTransform();
  });

  // Invert Colors
  elements.lightboxInvertBtn.addEventListener("click", () => {
    state.lightbox.isInverted = !state.lightbox.isInverted;
    elements.lightboxWrapper.classList.toggle("is-inverted", state.lightbox.isInverted);
    elements.lightboxInvertBtn.classList.toggle("active", state.lightbox.isInverted);
  });

  // Fade / Opacity Slider
  elements.lightboxFadeSlider.addEventListener("input", (e) => {
    state.lightbox.opacity = parseFloat(e.target.value);
    elements.lightboxWrapper.style.opacity = state.lightbox.opacity;
  });

  // Carousel & Slideshow
  elements.lightboxPrevBtn.addEventListener("click", () => navigateLightbox(-1));
  elements.lightboxNextBtn.addEventListener("click", () => navigateLightbox(1));
  elements.lightboxPlayBtn.addEventListener("click", toggleSlideshow);

  // Metadata Info Drawer Toggle
  elements.lightboxInfoBtn.addEventListener("click", toggleInfoDrawer);
  elements.closeInfoDrawer.addEventListener("click", () => elements.lightboxInfoDrawer.classList.add("hidden"));

  // Copy to Clipboard
  elements.lightboxCopyBtn.addEventListener("click", copyImageToClipboard);

  // Fullscreen Toggle
  elements.lightboxFullscreenBtn.addEventListener("click", toggleFullscreen);

  // Mouse Wheel Zoom
  elements.lightboxViewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    zoomAtPoint(factor, e.clientX, e.clientY);
  }, { passive: false });

  // Click on image to toggle zoom
  elements.lightboxImg.addEventListener("click", (e) => {
    if (state.lightbox.scale <= 1.05) {
      zoomAtPoint(2.5, e.clientX, e.clientY);
    } else {
      resetTransforms();
    }
  });

  // Pan & Drag Functionality
  elements.lightboxViewport.addEventListener("mousedown", (e) => {
    if (state.lightbox.scale <= 1.0) return;
    state.lightbox.isDragging = true;
    state.lightbox.startX = e.clientX - state.lightbox.translateX;
    state.lightbox.startY = e.clientY - state.lightbox.translateY;
    elements.lightboxViewport.classList.add("is-dragging");
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.lightbox.isDragging) return;
    e.preventDefault();
    state.lightbox.translateX = e.clientX - state.lightbox.startX;
    state.lightbox.translateY = e.clientY - state.lightbox.startY;
    applyTransform();
  });

  window.addEventListener("mouseup", () => {
    if (state.lightbox.isDragging) {
      state.lightbox.isDragging = false;
      elements.lightboxViewport.classList.remove("is-dragging");
    }
  });

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (elements.lightboxModal.classList.contains("hidden")) return;

    if (e.key === "Escape") {
      closeModal();
    } else if (e.key === "ArrowLeft") {
      navigateLightbox(-1);
    } else if (e.key === "ArrowRight") {
      navigateLightbox(1);
    } else if (e.key === "+" || e.key === "=") {
      zoomStep(1.3);
    } else if (e.key === "-" || e.key === "_") {
      zoomStep(1 / 1.3);
    } else if (e.key === "0") {
      resetTransforms();
    } else if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      toggleSlideshow();
    } else if (e.key === "r" || e.key === "R") {
      state.lightbox.rotation = (state.lightbox.rotation + 90) % 360;
      applyTransform();
    } else if (e.key === "h" || e.key === "H") {
      state.lightbox.flipH = state.lightbox.flipH === 1 ? -1 : 1;
      applyTransform();
    } else if (e.key === "i" || e.key === "I") {
      toggleInfoDrawer();
    } else if (e.key === "f" || e.key === "F") {
      toggleFullscreen();
    } else if (e.key === "c" || e.key === "C") {
      copyImageToClipboard();
    }
  });
}

function openLightbox(imagesList, index = 0, datasetKey = "cifar10") {
  if (!imagesList || imagesList.length === 0) return;
  state.lightbox.images = imagesList;
  state.lightbox.currentIndex = index;
  state.lightbox.currentTab = datasetKey;
  state.lightbox.isCrisp = (datasetKey === "cifar10" || datasetKey === "fashion_mnist");

  updateLightboxImage(false);
  updatePixelMode();
  updateInfoDrawer();
  resetTransforms();
  elements.lightboxModal.classList.remove("hidden");
}

function updateLightboxImage(withCrossfade = true) {
  const currentSrc = state.lightbox.images[state.lightbox.currentIndex];
  if (withCrossfade) {
    elements.lightboxImg.classList.add("fade-transition");
    setTimeout(() => {
      elements.lightboxImg.src = currentSrc;
      elements.lightboxImg.classList.remove("fade-transition");
    }, 120);
  } else {
    elements.lightboxImg.src = currentSrc;
  }

  elements.lightboxDownloadBtn.href = currentSrc;
  elements.lightboxCounter.textContent = `${state.lightbox.currentIndex + 1} / ${state.lightbox.images.length}`;
  updateInfoDrawer();
}

function updatePixelMode() {
  elements.lightboxWrapper.classList.toggle("crisp-pixel", state.lightbox.isCrisp);
  elements.lightboxWrapper.classList.toggle("smooth-pixel", !state.lightbox.isCrisp);
  elements.lightboxPixelToggleBtn.textContent = state.lightbox.isCrisp ? "👾 Crisp" : "✨ Smooth";
}

function navigateLightbox(direction) {
  const len = state.lightbox.images.length;
  if (len === 0) return;
  state.lightbox.currentIndex = (state.lightbox.currentIndex + direction + len) % len;
  updateLightboxImage(true);
}

function toggleSlideshow() {
  if (state.lightbox.isPlaying) {
    stopSlideshow();
  } else {
    startSlideshow();
  }
}

function startSlideshow() {
  state.lightbox.isPlaying = true;
  elements.lightboxPlayBtn.textContent = "⏸ Pause";
  elements.lightboxPlayBtn.classList.add("active");
  state.lightbox.playInterval = setInterval(() => {
    navigateLightbox(1);
  }, 2200);
}

function stopSlideshow() {
  state.lightbox.isPlaying = false;
  elements.lightboxPlayBtn.textContent = "▶ Slideshow";
  elements.lightboxPlayBtn.classList.remove("active");
  if (state.lightbox.playInterval) {
    clearInterval(state.lightbox.playInterval);
    state.lightbox.playInterval = null;
  }
}

function toggleInfoDrawer() {
  elements.lightboxInfoDrawer.classList.toggle("hidden");
  elements.lightboxInfoBtn.classList.toggle("active", !elements.lightboxInfoDrawer.classList.contains("hidden"));
}

function updateInfoDrawer() {
  const tab = state.lightbox.currentTab;
  const cfgData = state[tab];
  const resolution = tab === "celeba" ? "256 × 256 (RGB)" : (tab === "cifar10" ? "32 × 32 (RGB)" : "28 × 28 (Grayscale)");

  elements.infoDrawerContent.innerHTML = `
    <div class="info-row"><span class="label">Dataset</span><span class="val">${tab.toUpperCase()}</span></div>
    <div class="info-row"><span class="label">Resolution</span><span class="val">${resolution}</span></div>
    <div class="info-row"><span class="label">Sample Index</span><span class="val">#${state.lightbox.currentIndex + 1} of ${state.lightbox.images.length}</span></div>
    <div class="info-row"><span class="label">Seed</span><span class="val">${cfgData.seed}</span></div>
    <div class="info-row"><span class="label">DDIM Steps</span><span class="val">${cfgData.steps}</span></div>
    <div class="info-row"><span class="label">CFG Scale</span><span class="val">${cfgData.cfg}</span></div>
    <div class="info-row"><span class="label">Latency</span><span class="val">${cfgData.lastLatency ? cfgData.lastLatency.toFixed(2) + 's' : 'Cached'}</span></div>
    <div class="info-row"><span class="label">Conditioning</span><span class="val">${cfgData.lastCaption || 'None'}</span></div>
  `;
}

async function copyImageToClipboard() {
  const currentSrc = state.lightbox.images[state.lightbox.currentIndex];
  try {
    const res = await fetch(currentSrc);
    const blob = await res.blob();
    await navigator.clipboard.write([
      new ClipboardItem({ [blob.type]: blob })
    ]);
    const oldText = elements.lightboxCopyBtn.textContent;
    elements.lightboxCopyBtn.textContent = "✅ Copied!";
    setTimeout(() => elements.lightboxCopyBtn.textContent = oldText, 1500);
  } catch (err) {
    alert("Could not copy image automatically to clipboard: " + err.message);
  }
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    elements.lightboxContainer.requestFullscreen().catch(() => {});
    elements.lightboxFullscreenBtn.textContent = "⛶ Exit";
  } else {
    document.exitFullscreen().catch(() => {});
    elements.lightboxFullscreenBtn.textContent = "⛶";
  }
}

function zoomStep(factor) {
  const newScale = Math.min(Math.max(0.5, state.lightbox.scale * factor), 8.0);
  state.lightbox.scale = newScale;
  if (newScale <= 1.0) {
    state.lightbox.translateX = 0;
    state.lightbox.translateY = 0;
  }
  applyTransform();
}

function zoomAtPoint(factor, clientX, clientY) {
  const rect = elements.lightboxViewport.getBoundingClientRect();
  const oldScale = state.lightbox.scale;
  const newScale = Math.min(Math.max(0.5, oldScale * factor), 8.0);

  if (newScale === oldScale) return;

  const mouseX = clientX - (rect.left + rect.width / 2);
  const mouseY = clientY - (rect.top + rect.height / 2);

  state.lightbox.translateX -= (mouseX - state.lightbox.translateX) * (newScale / oldScale - 1);
  state.lightbox.translateY -= (mouseY - state.lightbox.translateY) * (newScale / oldScale - 1);
  state.lightbox.scale = newScale;

  if (newScale <= 1.0) {
    state.lightbox.translateX = 0;
    state.lightbox.translateY = 0;
  }

  applyTransform();
}

function resetTransforms() {
  state.lightbox.scale = 1.0;
  state.lightbox.rotation = 0;
  state.lightbox.flipH = 1;
  state.lightbox.opacity = 1.0;
  state.lightbox.translateX = 0;
  state.lightbox.translateY = 0;
  state.lightbox.isInverted = false;

  elements.lightboxFadeSlider.value = "1.0";
  elements.lightboxWrapper.style.opacity = "1.0";
  elements.lightboxWrapper.classList.remove("is-inverted");
  elements.lightboxInvertBtn.classList.remove("active");

  applyTransform();
}

function applyTransform() {
  const { scale, translateX, translateY, rotation, flipH } = state.lightbox;
  elements.lightboxWrapper.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale * flipH}, ${scale}) rotate(${rotation}deg)`;
  elements.lightboxZoomLevel.textContent = `${Math.round(scale * 100)}%`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function bindSlider(sliderId, displayId, onInput) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (!slider || !display) return;

  slider.addEventListener("input", (e) => {
    display.textContent = e.target.value;
    onInput(e.target.value);
  });
}

function clearActivePresets(presetContainerId) {
  const container = document.getElementById(presetContainerId);
  if (container) {
    container.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  }
}
