// Elements
const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const startBtn = document.getElementById("startBtn");
const textOutput = document.getElementById("textOutput");
const previewArea = document.getElementById("previewArea");
const loadingMsg = document.getElementById("loadingMsg");

// Open file chooser
chooseBtn.addEventListener("click", () => {
  fileInput.click();
});

// Show preview when a file is selected
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewArea.innerHTML = `<img src="${e.target.result}" alt="Preview" style="max-width:200px;border:1px solid #ccc;margin-top:10px;">`;
    };
    reader.readAsDataURL(file);
  }
});

// Extract text button
startBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first!");
    return;
  }

  // Show loading state
  startBtn.disabled = true;
  startBtn.textContent = "Extracting...";
  loadingMsg.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/extract", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Server error while extracting");

    const data = await response.json();
    textOutput.value = data.text || "No text found.";
  } catch (err) {
    console.error(err);
    alert("Failed to extract text. Check backend logs.");
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = "Extract Text ➜";
    loadingMsg.style.display = "none";
  }
});

// Copy extracted text to clipboard
function copyText() {
  if (!textOutput.value) return;
  navigator.clipboard.writeText(textOutput.value)
    .then(() => alert("Copied to clipboard!"))
    .catch(err => console.error("Copy failed:", err));
}
