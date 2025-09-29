// ATS Resume Checker JavaScript

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    initializeFileValidation();
    initializeFormValidation();
    initializeProgressBars();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File validation and preview
function initializeFileValidation() {
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            validateFiles(this.files);
            updateFilePreview(this.files);
        });
    }
}

// Validate selected files
function validateFiles(files) {
    const maxFiles = 10;
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['pdf', 'docx'];

    if (files.length > maxFiles) {
        showAlert('danger', `Maximum ${maxFiles} files allowed. Please remove some files.`);
        return false;
    }

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(fileExtension)) {
            showAlert('danger', `File "${file.name}" has unsupported format. Only PDF and DOCX files are allowed.`);
            return false;
        }

        if (file.size > maxSize) {
            showAlert('danger', `File "${file.name}" is too large. Maximum file size is 16MB.`);
            return false;
        }
    }

    return true;
}

// Update file preview list
function updateFilePreview(files) {
    const fileList = document.getElementById('fileList');
    const fileListItems = document.getElementById('fileListItems');

    if (!fileList || !fileListItems) return;

    fileListItems.innerHTML = '';

    if (files.length > 0) {
        fileList.style.display = 'block';

        Array.from(files).forEach((file, index) => {
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);
            const fileType = file.name.split('.').pop().toLowerCase();
            const iconClass = fileType === 'pdf' ? 'fas fa-file-pdf text-danger' : 'fas fa-file-word text-primary';

            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center fade-in';

            li.innerHTML = `
                <div>
                    <i class="${iconClass} me-2"></i>
                    <span class="file-name">${file.name}</span>
                </div>
                <div class="d-flex align-items-center">
                    <span class="badge bg-light text-dark me-2">${fileSize} MB</span>
                    <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;

            fileListItems.appendChild(li);
        });
    } else {
        fileList.style.display = 'none';
    }
}

// Remove file from selection (Note: This is simplified - actual implementation would need file list management)
function removeFile(index) {
    const fileInput = document.getElementById('files');
    if (fileInput) {
        // Note: HTML5 file input doesn't allow direct manipulation
        // This would require a more complex implementation with a file array
        showAlert('info', 'To remove files, please reselect your files without the unwanted ones.');
    }
}

// Form validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }

            form.classList.add('was-validated');
        });
    });

    // Upload form specific validation
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('files');
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                showAlert('danger', 'Please select at least one resume file to upload.');
                return;
            }

            if (fileInput && !validateFiles(fileInput.files)) {
                e.preventDefault();
                return;
            }

            // Show processing state
            showProcessingState();
        });
    }
}

// Show processing state
function showProcessingState() {
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.innerHTML = '<i class="spinner-border spinner-border-sm me-2"></i>Processing resumes...';
        uploadBtn.disabled = true;
    }

    // Show progress indicator
    showAlert('info', 'Processing resumes... This may take a few moments depending on file sizes.');
}

// Initialize progress bars
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');

    progressBars.forEach(bar => {
        const targetWidth = bar.getAttribute('data-width') || '0%';
        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 300);
    });
}

// Show alert message
function showAlert(type, message) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert alert at the top of main content
    const main = document.querySelector('main.container');
    if (main) {
        main.insertBefore(alertContainer, main.firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertContainer.remove();
        }, 5000);
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('success', 'Copied to clipboard!');
    }, function(err) {
        showAlert('danger', 'Failed to copy to clipboard.');
    });
}

// Export functions
function exportResults(format = 'csv') {
    window.location.href = `/export_results?format=${format}`;
}

// Search and filter functionality for results
function filterResults(category, searchTerm = '') {
    const tableRows = document.querySelectorAll(`#${category} tbody tr`);

    tableRows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const shouldShow = searchTerm === '' || text.includes(searchTerm.toLowerCase());
        row.style.display = shouldShow ? '' : 'none';
    });
}

// Initialize search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInputs = document.querySelectorAll('.search-input');
    searchInputs.forEach(input => {
        input.addEventListener('input', function() {
            const category = this.getAttribute('data-category');
            const searchTerm = this.value;
            filterResults(category, searchTerm);
        });
    });
});