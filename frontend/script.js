/* =========================================================
   AGRISENSE FRONTEND
   Connects the frontend to the FastAPI inference endpoint.
   ========================================================= */


const API_URL = "http://127.0.0.1:8000/predict";


// ==================== ELEMENTS ====================

const imageInput =
    document.getElementById("imageInput");

const browseButton =
    document.getElementById("browseButton");

const dropZone =
    document.getElementById("dropZone");

const uploadPlaceholder =
    document.getElementById("uploadPlaceholder");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const removeImage =
    document.getElementById("removeImage");

const fileName =
    document.getElementById("fileName");

const analyzeButton =
    document.getElementById("analyzeButton");

const analyzeText =
    document.getElementById("analyzeText");

const buttonSpinner =
    document.getElementById("buttonSpinner");

const resultsSection =
    document.getElementById("resultsSection");

const errorMessage =
    document.getElementById("errorMessage");

const errorText =
    document.getElementById("errorText");

const closeError =
    document.getElementById("closeError");

const newAnalysisButton =
    document.getElementById("newAnalysisButton");


// Result elements

const diseaseName =
    document.getElementById("diseaseName");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");

const classIndex =
    document.getElementById("classIndex");

const maskCoverage =
    document.getElementById("maskCoverage");

const imageDimensions =
    document.getElementById("imageDimensions");

const originalResultImage =
    document.getElementById("originalResultImage");

const segmentationImage =
    document.getElementById("segmentationImage");

const gradcamImage =
    document.getElementById("gradcamImage");


// ==================== STATE ====================

let selectedFile = null;


// ==================== FILE SELECTION ====================

browseButton.addEventListener(
    "click",
    () => {
        imageInput.click();
    }
);


imageInput.addEventListener(
    "change",
    (event) => {

        const file =
            event.target.files[0];

        if (file) {
            handleFile(file);
        }

    }
);


// ==================== DRAG & DROP ====================

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "drag-over"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "drag-over"
        );

    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "drag-over"
        );

        const file =
            event.dataTransfer.files[0];

        if (file) {
            handleFile(file);
        }

    }
);


// ==================== HANDLE FILE ====================

function handleFile(file) {

    hideError();

    if (!file.type.startsWith("image/")) {

        showError(
            "Please select a valid image file."
        );

        return;
    }


    selectedFile = file;

    fileName.textContent =
        file.name;


    const reader =
        new FileReader();


    reader.onload = (event) => {

        imagePreview.src =
            event.target.result;

        originalResultImage.src =
            event.target.result;

    };


    reader.readAsDataURL(file);


    uploadPlaceholder.classList.add(
        "hidden"
    );

    previewContainer.classList.remove(
        "hidden"
    );

    analyzeButton.disabled = false;


    imagePreview.onload =
        () => {

            imageDimensions.textContent =
                `${imagePreview.naturalWidth} × ${imagePreview.naturalHeight}`;

        };
}


// ==================== REMOVE IMAGE ====================

removeImage.addEventListener(
    "click",
    (event) => {

        event.stopPropagation();

        resetUpload();

    }
);


function resetUpload() {

    selectedFile = null;

    imageInput.value = "";

    imagePreview.src = "";

    originalResultImage.src = "";

    uploadPlaceholder.classList.remove(
        "hidden"
    );

    previewContainer.classList.add(
        "hidden"
    );

    analyzeButton.disabled = true;

}


// ==================== ANALYZE ====================

analyzeButton.addEventListener(
    "click",
    analyzeImage
);


async function analyzeImage() {

    if (!selectedFile) {
        return;
    }


    hideError();

    setLoading(true);


    const formData =
        new FormData();

    formData.append(
        "file",
        selectedFile
    );


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            let message =
                `Server returned status ${response.status}.`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {
                    message =
                        errorData.detail;
                }

            } catch (_) {
                // Keep default message.
            }

            throw new Error(message);
        }


        const result =
            await response.json();


        displayResults(result);


    } catch (error) {

        console.error(
            "AgriSense API error:",
            error
        );


        showError(
            getReadableError(error)
        );


    } finally {

        setLoading(false);

    }
}


// ==================== DISPLAY RESULTS ====================

function displayResults(result) {

    const confidence =
        Number(result.confidence) * 100;

    const coverage =
        Number(result.mask_coverage) * 100;


    diseaseName.textContent =
        formatDiseaseName(
            result.disease
        );


    confidenceValue.textContent =
        confidence.toFixed(2);


    classIndex.textContent =
        result.class_index;


    maskCoverage.textContent =
        `${coverage.toFixed(2)}%`;


    confidenceFill.style.width =
        `${Math.min(confidence, 100)}%`;


    /*
     * The API returns Base64-encoded PNG images.
     *
     * Prefixing the Base64 data with:
     * data:image/png;base64,
     * allows the browser to display it directly.
     */

    segmentationImage.src =
        `data:image/png;base64,${result.segmentation_mask}`;


    gradcamImage.src =
        `data:image/png;base64,${result.gradcam}`;


    resultsSection.classList.remove(
        "hidden"
    );


    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// ==================== DISEASE NAME ====================

function formatDiseaseName(name) {

    if (!name) {
        return "Unknown";
    }


    return name
        .replaceAll("_", " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(
            /\b\w/g,
            character =>
                character.toUpperCase()
        );

}


// ==================== LOADING STATE ====================

function setLoading(isLoading) {

    analyzeButton.disabled =
        isLoading || !selectedFile;


    if (isLoading) {

        buttonSpinner.classList.remove(
            "hidden"
        );

        analyzeText.textContent =
            "Analyzing...";

    } else {

        buttonSpinner.classList.add(
            "hidden"
        );

        analyzeText.textContent =
            "Analyze Image";

    }

}


// ==================== ERROR HANDLING ====================

function showError(message) {

    errorText.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorMessage.classList.add(
        "hidden"
    );

}


closeError.addEventListener(
    "click",
    hideError
);


// ==================== READABLE ERRORS ====================

function getReadableError(error) {

    if (
        error instanceof TypeError &&
        error.message.includes("fetch")
    ) {

        return (
            "The AgriSense API could not be reached. " +
            "Make sure the FastAPI server is running."
        );

    }


    return (
        error.message ||
        "An unexpected error occurred during analysis."
    );

}


// ==================== NEW ANALYSIS ====================

newAnalysisButton.addEventListener(
    "click",
    () => {

        resultsSection.classList.add(
            "hidden"
        );

        resetUpload();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);