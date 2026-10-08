const audioFile = document.getElementById("audioFile");
const selectButton = document.getElementById("selectButton");
const analyzeButton = document.getElementById("analyzeButton");

const filename = document.getElementById("filename");

const loading = document.getElementById("loading");
const error = document.getElementById("error");
const result = document.getElementById("result");

selectButton.addEventListener("click", () => {
    audioFile.click();
});


audioFile.addEventListener("change", () => {

    if (audioFile.files.length === 0) {
        filename.textContent = "";
        analyzeButton.disabled = true;
        return;
    }

    const file = audioFile.files[0];

    filename.textContent = file.name;

    analyzeButton.disabled = false;

    result.style.display = "none";
    error.style.display = "none";
});


analyzeButton.addEventListener("click", async () => {

    if (audioFile.files.length === 0) {
        return;
    }

    const file = audioFile.files[0];

    const formData = new FormData();

    formData.append("audio", file);


    analyzeButton.disabled = true;

    loading.style.display = "block";
    result.style.display = "none";
    error.style.display = "none";


    try {

        const response = await fetch(
            "/api/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!data.success) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }


        displayResult(data);


    } catch (err) {

        error.textContent = err.message;
        error.style.display = "block";

    } finally {

        loading.style.display = "none";
        analyzeButton.disabled = false;

    }

});


function displayResult(data) {

    const prediction =
        document.getElementById("prediction");

    prediction.textContent = data.prediction;

    prediction.className =
        "prediction " +
        data.prediction.toLowerCase();


    document.getElementById("confidence").textContent =
        "Confidence: " +
        data.confidence +
        "%";


    document.getElementById("realValue").textContent =
        data.real_probability + "%";


    document.getElementById("fakeValue").textContent =
        data.fake_probability + "%";


    document.getElementById("realBar").style.width =
        data.real_probability + "%";


    document.getElementById("fakeBar").style.width =
        data.fake_probability + "%";


    document.getElementById("resultFile").textContent =
        data.filename;


    document.getElementById("duration").textContent =
        data.duration + " seconds";


    document.getElementById("segments").textContent =
        data.segments_analyzed;


    document.getElementById("realSegments").textContent =
        data.real_segments;


    document.getElementById("fakeSegments").textContent =
        data.fake_segments;


    result.style.display = "block";
}