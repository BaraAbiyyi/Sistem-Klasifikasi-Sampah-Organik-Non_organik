// JavaScript untuk halaman deteksi

let stream = null;

// Tab switching
function switchTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tab + '-tab').classList.add('active');
    event.target.classList.add('active');
    
    // Stop webcam if switching away
    if (tab !== 'webcam' && stream) {
        stopWebcam();
    }
}

// File upload handling
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const selectImageBtn = document.getElementById('selectImageBtn');

// Prevent double click - hanya handle click pada button atau area uploadBox (tapi bukan button)
if (uploadBox) {
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#00796B';
        uploadBox.style.background = '#48A999';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '#48A999';
        uploadBox.style.background = '#FAFAFA';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#48A999';
        uploadBox.style.background = '#FAFAFA';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle click pada uploadBox area (tapi bukan button)
    uploadBox.addEventListener('click', (e) => {
        // Jika yang diklik adalah button, jangan trigger (biarkan button handler yang handle)
        if (e.target.closest('button')) {
            return;
        }
        // Jika klik di area uploadBox (bukan button), trigger file input
        fileInput.click();
    });
}

// Handle click pada button secara terpisah
if (selectImageBtn) {
    selectImageBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent event bubbling ke uploadBox
        fileInput.click();
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Silakan pilih file gambar!');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        document.getElementById('resultSection').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
}

function resetDetection() {
    removeImage();
}

// Predict image
async function predictImage() {
    if (!fileInput.files[0]) {
        alert('Silakan pilih gambar terlebih dahulu!');
        return;
    }
    
    const loading = document.getElementById('loading');
    loading.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Cek apakah menggunakan YOLO
    const useYOLO = document.getElementById('useYOLO');
    if (useYOLO) {
        formData.append('use_yolo', useYOLO.checked);
    }
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        loading.style.display = 'none';
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        displayResult(data.result);
        
    } catch (error) {
        loading.style.display = 'none';
        alert('Error: ' + error.message);
        console.error('Error:', error);
    }
}

function displayResult(result) {
    const resultIcon = document.getElementById('resultIcon');
    const resultClass = document.getElementById('resultClass');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const processingTime = document.getElementById('processingTime');
    const resultSection = document.getElementById('resultSection');
    const resultCard = resultSection.querySelector('.result-card');
    const bboxContainer = document.getElementById('bboxImageContainer');
    const bboxImage = document.getElementById('bboxImage');
    const wasteInfoCard = document.getElementById('wasteInfoCard');
    const detectionSummary = document.getElementById('detectionSummary');
    const multipleDetections = document.getElementById('multipleDetections');
    const detectionsList = document.getElementById('detectionsList');
    
    // Tampilkan multiple detections jika ada
    if (result.has_bbox && result.detections && result.detections.length > 0) {
        // Tampilkan summary
        if (result.detection_summary) {
            document.getElementById('totalObjects').textContent = result.detection_summary.total;
            document.getElementById('organicCount').textContent = result.detection_summary.organic;
            document.getElementById('nonOrganicCount').textContent = result.detection_summary.non_organic;
            detectionSummary.style.display = 'block';
        }
        
        // Tampilkan list semua deteksi
        detectionsList.innerHTML = '';
        result.detections.forEach((det, index) => {
            const detItem = document.createElement('div');
            detItem.className = 'detection-item-detail';
            detItem.innerHTML = `
                <div class="det-item-icon">
                    ${det.class === 'Organik' ? '<i class="fas fa-leaf"></i>' : '<i class="fas fa-recycle"></i>'}
                </div>
                <div class="det-item-info">
                    <h4>Objek ${index + 1}: ${det.class}</h4>
                    <p>Confidence: <strong>${det.confidence}%</strong></p>
                    <p class="det-bbox">Bounding Box: [${det.bbox.join(', ')}]</p>
                </div>
                <div class="det-item-confidence">
                    <div class="confidence-badge confidence-${det.class === 'Organik' ? 'success' : 'warning'}">
                        ${det.confidence}%
                    </div>
                </div>
            `;
            detectionsList.appendChild(detItem);
        });
        multipleDetections.style.display = 'block';
        
        // Tentukan kelas utama berdasarkan yang paling banyak
        const organicCount = result.detection_summary.organic;
        const nonOrganicCount = result.detection_summary.non_organic;
        
        if (organicCount >= nonOrganicCount) {
            resultIcon.textContent = '🍃';
            resultCard.style.borderColor = '#8BC34A';
            resultClass.textContent = `Sampah Organik (${organicCount} objek)`;
            resultClass.style.color = '#8BC34A';
        } else {
            resultIcon.textContent = '♻️';
            resultCard.style.borderColor = '#FB8C00';
            resultClass.textContent = `Sampah Non-Organik (${nonOrganicCount} objek)`;
            resultClass.style.color = '#FB8C00';
        }
        
        // Confidence dari deteksi terbaik
        const maxConfidence = Math.max(...result.detections.map(d => d.confidence));
        confidenceFill.style.width = maxConfidence + '%';
        confidenceFill.textContent = maxConfidence + '%';
        confidenceText.textContent = `Tingkat Keyakinan Tertinggi: ${maxConfidence}%`;
        
    } else {
        // Single detection (tanpa YOLO)
        detectionSummary.style.display = 'none';
        multipleDetections.style.display = 'none';
        
        if (result.class === 'Organik') {
            resultIcon.textContent = '🍃';
            resultCard.style.borderColor = '#8BC34A';
            resultClass.textContent = 'Sampah Organik';
            resultClass.style.color = '#8BC34A';
        } else {
            resultIcon.textContent = '♻️';
            resultCard.style.borderColor = '#FB8C00';
            resultClass.textContent = 'Sampah Non-Organik';
            resultClass.style.color = '#FB8C00';
        }
        
        const confidence = result.confidence;
        confidenceFill.style.width = confidence + '%';
        confidenceFill.textContent = confidence + '%';
        confidenceText.textContent = `Tingkat Keyakinan: ${confidence}%`;
    }
    
    if (processingTime) {
        processingTime.textContent = result.processing_time || 'N/A';
    }
    
    // Tampilkan informasi detail jika ada
    if (result.info) {
        displayWasteInfo(result.info);
        wasteInfoCard.style.display = 'block';
    } else {
        wasteInfoCard.style.display = 'none';
    }
    
    // Tampilkan bounding box image jika ada
    if (result.bbox_image) {
        bboxImage.src = `/static/uploads/${result.bbox_image}`;
        bboxContainer.style.display = 'block';
    } else {
        bboxContainer.style.display = 'none';
    }
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayWasteInfo(info) {
    // Description
    document.getElementById('wasteDescription').textContent = info.description;
    
    // Examples
    const examplesList = document.getElementById('wasteExamples');
    examplesList.innerHTML = '';
    info.examples.forEach(example => {
        const li = document.createElement('li');
        li.textContent = example;
        examplesList.appendChild(li);
    });
    
    // Recycling status
    const recyclingBadge = document.getElementById('recyclingBadge');
    const recyclingInfo = document.getElementById('recyclingInfo');
    
    if (info.recyclable) {
        recyclingBadge.innerHTML = '<span class="badge-recyclable"><i class="fas fa-check-circle"></i> Dapat Didaur Ulang</span>';
        recyclingBadge.className = 'recycling-badge recyclable';
    } else {
        recyclingBadge.innerHTML = '<span class="badge-not-recyclable"><i class="fas fa-times-circle"></i> Tidak Dapat Didaur Ulang (Konvensional)</span>';
        recyclingBadge.className = 'recycling-badge not-recyclable';
    }
    
    recyclingInfo.textContent = info.recycling_info;
    
    // Disposal tips
    const disposalTipsList = document.getElementById('disposalTips');
    disposalTipsList.innerHTML = '';
    info.disposal_tips.forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        disposalTipsList.appendChild(li);
    });
    
    // Benefits
    document.getElementById('wasteBenefits').textContent = info.benefits;
}

// Webcam functions
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment' // Use back camera if available
            } 
        });
        
        const webcam = document.getElementById('webcam');
        webcam.srcObject = stream;
        
        document.getElementById('startWebcam').style.display = 'none';
        document.getElementById('captureBtn').style.display = 'inline-flex';
        document.getElementById('stopWebcam').style.display = 'inline-flex';
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Tidak dapat mengakses webcam. Pastikan izin kamera sudah diberikan.');
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    const webcam = document.getElementById('webcam');
    webcam.srcObject = null;
    
    document.getElementById('startWebcam').style.display = 'inline-flex';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('stopWebcam').style.display = 'none';
}

function captureImage() {
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);
    
    canvas.toBlob(async (blob) => {
        const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
        
        // Set file to input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Copy YOLO setting dari webcam ke upload
        const useYOLOWebcam = document.getElementById('useYOLOWebcam');
        const useYOLO = document.getElementById('useYOLO');
        if (useYOLOWebcam && useYOLO) {
            useYOLO.checked = useYOLOWebcam.checked;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadBox.style.display = 'none';
            previewSection.style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';
        };
        reader.readAsDataURL(file);
        
        // Switch to upload tab
        switchTab('upload');
    }, 'image/jpeg', 0.95);
}

// Video file handling
const videoInput = document.getElementById('videoInput');
const videoUploadBox = document.getElementById('videoUploadBox');
const videoPreviewSection = document.getElementById('videoPreviewSection');
const videoResultSection = document.getElementById('videoResultSection');
const selectVideoBtn = document.getElementById('selectVideoBtn');

// Initialize video handlers when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (videoInput && videoUploadBox) {
        // Drag and drop handlers
        videoUploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoUploadBox.style.borderColor = '#00796B';
            videoUploadBox.style.background = '#48A999';
        });

        videoUploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoUploadBox.style.borderColor = '#48A999';
            videoUploadBox.style.background = '#FAFAFA';
        });

        videoUploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoUploadBox.style.borderColor = '#48A999';
            videoUploadBox.style.background = '#FAFAFA';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                handleVideoFile(files[0]);
            } else {
                alert('Silakan pilih file video!');
            }
        });

        // Handle click pada uploadBox area (tapi bukan button atau input)
        videoUploadBox.addEventListener('click', (e) => {
            // Jika yang diklik adalah button, input, atau child dari button, jangan trigger
            if (e.target.closest('button') || e.target.closest('input') || e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') {
                return;
            }
            // Jika klik di area uploadBox (bukan button), trigger file input
            if (videoInput) {
                videoInput.click();
            }
        });
    }

    // Handle click pada button secara terpisah - HANYA button handler, tidak ada onclick di HTML
    if (selectVideoBtn && videoInput) {
        // Remove any existing listeners by cloning
        const newBtn = selectVideoBtn.cloneNode(true);
        selectVideoBtn.parentNode.replaceChild(newBtn, selectVideoBtn);
        const freshBtn = document.getElementById('selectVideoBtn');
        
        freshBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation(); // Prevent event bubbling ke uploadBox
            console.log('Select video button clicked');
            if (videoInput) {
                videoInput.click();
            }
        }, { once: false, passive: false });
    }

    // Handle file input change
    if (videoInput) {
        videoInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const file = e.target.files[0];
                if (file.type.startsWith('video/')) {
                    handleVideoFile(file);
                } else {
                    alert('Silakan pilih file video!');
                    videoInput.value = ''; // Reset input
                }
            }
        });
    }
});

function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
        alert('Silakan pilih file video!');
        return;
    }
    
    const fileSize = (file.size / (1024 * 1024)).toFixed(2);
    document.getElementById('videoFileName').textContent = file.name;
    document.getElementById('videoFileSize').textContent = fileSize + ' MB';
    
    videoUploadBox.style.display = 'none';
    videoPreviewSection.style.display = 'block';
    videoResultSection.style.display = 'none';
}

function removeVideo() {
    videoInput.value = '';
    videoUploadBox.style.display = 'block';
    videoPreviewSection.style.display = 'none';
    videoResultSection.style.display = 'none';
}

function resetVideo() {
    removeVideo();
}

async function processVideo() {
    if (!videoInput.files[0]) {
        alert('Silakan pilih video terlebih dahulu!');
        return;
    }
    
    const loading = document.getElementById('loading');
    loading.style.display = 'flex';
    loading.querySelector('p').textContent = 'Memproses video... (ini mungkin memakan waktu)';
    
    const formData = new FormData();
    formData.append('file', videoInput.files[0]);
    
    try {
        const response = await fetch('/detect_video', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        loading.style.display = 'none';
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Tampilkan hasil dengan detail lengkap
        const summary = data.summary;
        const totalDetections = data.total_detections || (summary.Organik + summary['Non-Organik']);
        const organicPercent = totalDetections > 0 ? ((summary.Organik / totalDetections) * 100).toFixed(1) : 0;
        const nonOrganicPercent = totalDetections > 0 ? ((summary['Non-Organik'] / totalDetections) * 100).toFixed(1) : 0;
        
        const summaryHTML = `
            <div class="video-result-header">
                <h4><i class="fas fa-chart-bar"></i> Ringkasan Deteksi Video</h4>
            </div>
            <div class="video-stats-grid">
                <div class="video-stat-card">
                    <div class="video-stat-icon organic">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <div class="video-stat-content">
                        <h3>${summary.Organik}</h3>
                        <p>Deteksi Organik</p>
                        <span class="video-stat-percent">${organicPercent}%</span>
                    </div>
                </div>
                <div class="video-stat-card">
                    <div class="video-stat-icon non-organic">
                        <i class="fas fa-recycle"></i>
                    </div>
                    <div class="video-stat-content">
                        <h3>${summary['Non-Organik']}</h3>
                        <p>Deteksi Non-Organik</p>
                        <span class="video-stat-percent">${nonOrganicPercent}%</span>
                    </div>
                </div>
                <div class="video-stat-card">
                    <div class="video-stat-icon info">
                        <i class="fas fa-cube"></i>
                    </div>
                    <div class="video-stat-content">
                        <h3>${totalDetections}</h3>
                        <p>Total Deteksi</p>
                        <span class="video-stat-percent">${data.total_frames} frames</span>
                    </div>
                </div>
            </div>
            <div class="video-details">
                <h5><i class="fas fa-info-circle"></i> Informasi Video:</h5>
                <div class="video-details-grid">
                    <div class="video-detail-item">
                        <span class="detail-label"><i class="fas fa-film"></i> Resolusi:</span>
                        <span class="detail-value">${data.video_info.width}x${data.video_info.height}</span>
                    </div>
                    <div class="video-detail-item">
                        <span class="detail-label"><i class="fas fa-clock"></i> Durasi:</span>
                        <span class="detail-value">${data.video_info.duration.toFixed(2)} detik</span>
                    </div>
                    <div class="video-detail-item">
                        <span class="detail-label"><i class="fas fa-tachometer-alt"></i> FPS:</span>
                        <span class="detail-value">${data.video_info.fps}</span>
                    </div>
                    <div class="video-detail-item">
                        <span class="detail-label"><i class="fas fa-stopwatch"></i> Processing Time:</span>
                        <span class="detail-value">${data.processing_time.toFixed(2)} detik</span>
                    </div>
                </div>
            </div>
            <div class="video-main-result">
                <div class="video-result-badge ${data.main_class === 'Organik' ? 'organic' : 'non-organic'}">
                    <i class="fas fa-${data.main_class === 'Organik' ? 'leaf' : 'recycle'}"></i>
                    <div>
                        <h4>Kelas Utama: ${data.main_class}</h4>
                        <p>Confidence: ${data.main_confidence}%</p>
                    </div>
                </div>
            </div>
            ${data.detection_id ? `
            <div class="video-saved-notice">
                <i class="fas fa-check-circle"></i> Hasil deteksi telah disimpan ke history (ID: ${data.detection_id})
            </div>
            ` : ''}
        `;
        document.getElementById('videoSummary').innerHTML = summaryHTML;
        
        // Tampilkan video hasil
        const resultVideo = document.getElementById('resultVideo');
        const videoPlayerContainer = document.querySelector('.video-player-container');
        
        if (!resultVideo) {
            console.error('Video element not found!');
            alert('Error: Video player element tidak ditemukan');
            return;
        }
        
        // Gunakan static path langsung (lebih reliable)
        const videoUrl = `/static/uploads/${data.output_video}`;
        console.log('Loading video from:', videoUrl);
        console.log('Output video filename:', data.output_video);
        
        // Show video container immediately
        resultVideo.style.display = 'block';
        resultVideo.style.visibility = 'visible';
        if (videoPlayerContainer) {
            videoPlayerContainer.style.display = 'block';
            videoPlayerContainer.style.visibility = 'visible';
        }
        
        // Set source dengan cara yang lebih reliable
        // Clear dulu, lalu set baru
        resultVideo.removeAttribute('src');
        resultVideo.load();
        
        // Set source baru setelah clear
        setTimeout(() => {
            resultVideo.src = videoUrl;
            
            // Add error handler dengan informasi lebih detail
            let errorHandled = false;
            resultVideo.onerror = function() {
                if (errorHandled) return; // Prevent multiple error handling
                
                const error = resultVideo.error;
                let errorMsg = 'Unknown error';
                let errorCode = null;
                
                if (error) {
                    errorCode = error.code;
                    switch(error.code) {
                        case error.MEDIA_ERR_ABORTED:
                            errorMsg = 'Video loading aborted';
                            break;
                        case error.MEDIA_ERR_NETWORK:
                            errorMsg = 'Network error while loading video';
                            break;
                        case error.MEDIA_ERR_DECODE:
                            errorMsg = 'Video decoding error - codec tidak didukung atau file corrupt';
                            break;
                        case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                            errorMsg = 'Video format tidak didukung browser';
                            break;
                    }
                }
                
                console.error('Error loading video from:', videoUrl);
                console.error('Video error code:', errorCode);
                console.error('Error message:', errorMsg);
                
                // Show user-friendly error message
                const videoContainer = document.querySelector('.video-player-container');
                if (videoContainer) {
                    // Remove existing error message if any
                    const existingError = videoContainer.querySelector('.video-error-message');
                    if (existingError) {
                        existingError.remove();
                    }
                    
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'alert alert-warning video-error-message';
                    errorDiv.style.marginTop = '1rem';
                    errorDiv.innerHTML = `
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>Peringatan:</strong> Video tidak dapat diputar di browser. ${errorMsg}
                        <br><small>File video mungkin menggunakan codec yang tidak didukung. 
                        Silakan <a href="${videoUrl}" download="${data.output_video}" style="color: var(--primary-main); font-weight: bold;">download video</a> untuk melihat hasilnya.</small>
                    `;
                    videoContainer.appendChild(errorDiv);
                }
                
                // Try route /video/ sebagai fallback (hanya sekali)
                if (!errorHandled) {
                    errorHandled = true;
                    console.log('Trying route /video/ as fallback...');
                    const fallbackUrl = `/video/${data.output_video}`;
                    resultVideo.src = fallbackUrl;
                    resultVideo.load();
                }
            };
            
            // Add success handlers
            resultVideo.onloadedmetadata = function() {
                console.log('Video metadata loaded successfully');
                console.log('Video duration:', resultVideo.duration);
                console.log('Video dimensions:', resultVideo.videoWidth, 'x', resultVideo.videoHeight);
                resultVideo.style.display = 'block';
            };
            
            resultVideo.oncanplay = function() {
                console.log('Video can play');
                resultVideo.style.display = 'block';
            };
            
            resultVideo.onloadeddata = function() {
                console.log('Video data loaded');
                resultVideo.style.display = 'block';
            };
            
            // Load video
            resultVideo.load();
        }, 100);
        
        // Download link - gunakan static path
        const downloadLink = document.getElementById('downloadVideo');
        if (downloadLink) {
            downloadLink.href = videoUrl;
            downloadLink.download = data.output_video;
        }
        
        // Test video availability
        fetch(videoUrl, { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    console.log('Video file exists and is accessible');
                } else {
                    console.warn('Video file may not be accessible:', response.status);
                }
            })
            .catch(err => {
                console.warn('Error checking video file:', err);
            });
        
        videoPreviewSection.style.display = 'none';
        videoResultSection.style.display = 'block';
        
        // Ensure video container is visible
        const videoContainer = document.querySelector('.video-player-container');
        if (videoContainer) {
            videoContainer.style.display = 'block';
            videoContainer.style.visibility = 'visible';
        }
        
        // Scroll ke hasil setelah video dimuat
        setTimeout(() => {
            videoResultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 300);
        
    } catch (error) {
        loading.style.display = 'none';
        alert('Error: ' + error.message);
        console.error('Error:', error);
    }
}

