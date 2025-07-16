const dropArea = document.getElementById("drop-area");
const fileElem = document.getElementById("fileElem");
const uploadStatus = document.getElementById("upload-status");
const predictBtn = document.getElementById("predict-button");
const sampleBtn = document.getElementById("sample-button");
const evaluationContainer = document.getElementById("evaluation-container");

let evaluationMetrics = null;
let plotImageUrl = null;

dropArea.addEventListener("click", () => fileElem.click());

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    uploadFile(e.dataTransfer.files[0]);
  }
});

fileElem.addEventListener("change", (e) => {
  if (fileElem.files.length) {
    uploadFile(fileElem.files[0]);
  }
});

sampleBtn.addEventListener("click", () => {
  fetch('/sample_csv')
    .then(res => res.text())
    .then(csvText => {
      // Create a blob & file from sample CSV text
      const file = new File([csvText], "sample.csv", { type: "text/csv" });
      uploadFile(file);
    });
});

function uploadFile(file) {
  uploadStatus.textContent = "Uploading...";
  evaluationContainer.innerHTML = "";
  predictBtn.style.display = "none";

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Upload failed");
      }
      return res.json();
    })
    .then((data) => {
      uploadStatus.textContent = data.message;
      evaluationMetrics = data.metrics;
      plotImageUrl = data.plot_url;
      predictBtn.style.display = "inline-block";
      evaluationContainer.innerHTML = "";
    })
    .catch((err) => {
      uploadStatus.textContent = "Error: " + err.message;
    });
}

predictBtn.addEventListener("click", () => {
  showEvaluationAndForm();
});

function showEvaluationAndForm() {
  evaluationContainer.innerHTML = `
    <h3>Evaluation Metrics:</h3>
    <pre>${JSON.stringify(evaluationMetrics, null, 2)}</pre>
    <img src="data:image/png;base64,${plotImageUrl}" alt="Evaluation Chart" style="max-width: 100%; height: auto;"/>
    ${buildUserInputFormHTML()}
  `;

  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = Number(value);
    });

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || "Prediction failed");
        }
        return res.json();
      })
      .then((res) => {
        document.getElementById("prediction-result").innerText =
          "Predicted Placement Package: ₹ " + res.prediction;
      })
      .catch((err) => {
        document.getElementById("prediction-result").innerText =
          "Error: " + err.message;
      });
  });
}

function buildUserInputFormHTML() {
  return `
    <h3>Enter your details to predict placement package:</h3>
    <form id="prediction-form" class="row g-3">
      <div class="col-md-3">
        <input type="number" step="0.01" name="cgpa" placeholder="CGPA" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="0.01" name="12th_marks" placeholder="12th Marks" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="0.01" name="10th_marks" placeholder="10th Marks" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="1" min="1" max="10" name="communication_skill" placeholder="Communication Skill (1-10)" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="1" min="1" max="10" name="programming_skill" placeholder="Programming Skill (1-10)" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="1" min="0" name="number_of_internships" placeholder="Number of Internships" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="1" min="0" name="number_of_projects" placeholder="Number of Projects" class="form-control" required />
      </div>
      <div class="col-md-3">
        <input type="number" step="1" min="0" name="number_of_backlog" placeholder="Number of Backlogs" class="form-control" required />
      </div>
      <div class="col-12">
        <button type="submit" class="btn btn-primary">Predict Placement Package</button>
      </div>
    </form>
    <div id="prediction-result" class="mt-3 fw-bold"></div>
  `;
}
