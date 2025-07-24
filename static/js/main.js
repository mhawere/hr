// Updated HR Application JavaScript with Photo Upload and Department Management

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    initializeFormValidation();
    initializeDataTables();
    initializeModals();
    initializeNavigation();
    initializePhotoUpload();
    initializeBiometricIdValidation(); // NEW: Properly organized biometric validation
}

function initializePhotoUpload() {
    const photoInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
    
    photoInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            previewPhoto(e.target);
        });
    });
}

function previewPhoto(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Create or update preview
            let preview = input.parentNode.querySelector('.photo-preview');
            if (!preview) {
                preview = document.createElement('div');
                preview.className = 'photo-preview';
                input.parentNode.appendChild(preview);
            }
            
            preview.innerHTML = `
                <img src="${e.target.result}" alt="Photo Preview" style="max-width: 150px; max-height: 150px; border-radius: 8px; margin-top: 10px;">
                <button type="button" onclick="removePhotoPreview(this)" class="btn btn-sm btn-secondary" style="margin-left: 10px;">Remove</button>
            `;
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

function removePhotoPreview(button) {
    const preview = button.parentNode;
    const input = preview.parentNode.querySelector('input[type="file"]');
    input.value = '';
    preview.remove();
}

// NEW: Organized biometric ID validation function
function initializeBiometricIdValidation() {
    const biometricIdField = document.getElementById('biometric_id');
    if (!biometricIdField) return; // Exit if field doesn't exist on this page
    
    // Real-time input validation
    biometricIdField.addEventListener('input', function() {
        handleBiometricIdInput(this);
    });
    
    // Validation on blur
    biometricIdField.addEventListener('blur', function() {
        validateBiometricIdOnBlur(this);
    });
}

function handleBiometricIdInput(field) {
    // Convert to uppercase for consistency
    field.value = field.value.toUpperCase();
    
    // Remove any non-alphanumeric characters
    field.value = field.value.replace(/[^A-Z0-9]/g, '');
    
    // Validate length and pattern
    if (field.value.length > 0) {
        if (field.value.length < 3) {
            showFieldError(field, 'Biometric ID must be at least 3 characters');
        } else if (field.value.length > 20) {
            field.value = field.value.substring(0, 20);
            clearFieldError(field);
        } else {
            clearFieldError(field);
        }
    } else {
        clearFieldError(field);
    }
}

function validateBiometricIdOnBlur(field) {
    if (field.value && field.value.length < 3) {
        showFieldError(field, 'Biometric ID must be at least 3 characters');
    }
}

function validateBiometricIdInForm(field) {
    if (!field || !field.value.trim()) return true; // Optional field
    
    const biometricId = field.value.trim();
    const biometricPattern = /^[A-Za-z0-9]{3,20}$/;
    
    if (!biometricPattern.test(biometricId)) {
        showFieldError(field, 'Biometric ID must be 3-20 alphanumeric characters only');
        return false;
    } else {
        clearFieldError(field);
        return true;
    }
}

function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
            }
        });
        
        const inputs = form.querySelectorAll('input[required], select[required]');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
        });
    });
}

function validateForm(form) {
    let isValid = true;
    const requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });
    
    // NEW: Add biometric ID validation to form validation
    const biometricIdField = form.querySelector('#biometric_id');
    if (biometricIdField && !validateBiometricIdInForm(biometricIdField)) {
        isValid = false;
    }
    
    return isValid;
}

function validateField(field) {
    const value = field.value.trim();
    const fieldGroup = field.closest('.form-group');
    let isValid = true;
    
    const existingError = fieldGroup.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    if (field.hasAttribute('required') && !value) {
        showFieldError(field, 'This field is required');
        isValid = false;
    }
    
    if (field.type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            showFieldError(field, 'Please enter a valid email address');
            isValid = false;
        }
    }
    
    if (field.type === 'tel' && value) {
        const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
        if (!phoneRegex.test(value.replace(/\s/g, ''))) {
            showFieldError(field, 'Please enter a valid phone number');
            isValid = false;
        }
    }
    
    if (isValid) {
        field.classList.remove('error');
        field.classList.add('valid');
    } else {
        field.classList.remove('valid');
        field.classList.add('error');
    }
    
    return isValid;
}

// NEW: Helper functions for error handling
function showFieldError(field, message) {
    // Clear any existing errors first
    clearFieldError(field);
    
    const fieldGroup = field.closest('.form-group');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    fieldGroup.appendChild(errorDiv);
    
    // Add error styling to field
    field.classList.add('error');
    field.classList.remove('valid');
}

function clearFieldError(field) {
    const fieldGroup = field.closest('.form-group');
    const existingError = fieldGroup.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Remove error styling
    field.classList.remove('error');
}

function initializeDataTables() {
    const tables = document.querySelectorAll('.data-table');
    
    tables.forEach(table => {
        addTableSearch(table);
        addTableSorting(table);
    });
}

function addTableSearch(table) {
    const tableContainer = table.closest('.table-container');
    const tableHeader = tableContainer.querySelector('.table-header');
    
    if (tableHeader.querySelector('.table-search')) return; // Already exists
    
    const searchDiv = document.createElement('div');
    searchDiv.className = 'table-search';
    searchDiv.innerHTML = `
        <input type="text" placeholder="Search..." class="search-input">
        <i class="fas fa-search"></i>
    `;
    
    tableHeader.appendChild(searchDiv);
    
    const searchInput = searchDiv.querySelector('.search-input');
    const rows = table.querySelectorAll('tbody tr');
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const shouldShow = text.includes(searchTerm);
            row.style.display = shouldShow ? '' : 'none';
        });
    });
}

function addTableSorting(table) {
    const headers = table.querySelectorAll('th');
    
    headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', function() {
            sortTable(table, index);
        });
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isNumeric = isColumnNumeric(table, columnIndex);
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex].textContent.trim();
        const bText = b.cells[columnIndex].textContent.trim();
        
        if (isNumeric) {
            return parseFloat(aText) - parseFloat(bText);
        } else {
            return aText.localeCompare(bText);
        }
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

function isColumnNumeric(table, columnIndex) {
    const rows = table.querySelectorAll('tbody tr');
    let numericCount = 0;
    let totalCount = 0;
    
    for (let i = 0; i < Math.min(rows.length, 5); i++) {
        const cellText = rows[i].cells[columnIndex].textContent.trim();
        if (cellText) {
            totalCount++;
            if (!isNaN(parseFloat(cellText))) {
                numericCount++;
            }
        }
    }
    
    return numericCount > totalCount * 0.7;
}

function initializeModals() {
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });
    
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const openModal = document.querySelector('.modal[style*="block"]');
            if (openModal) {
                openModal.style.display = 'none';
            }
        }
    });
}

function initializeNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && currentPath.includes(href)) {
            link.classList.add('active');
        }
    });
}

// Department Management Functions
function deleteDepartment(deptId) {
    if (confirm('Are you sure you want to delete this department?')) {
        fetch(`/departments/delete/${deptId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).then(response => {
            if (response.ok) {
                location.reload();
            } else {
                alert('Error deleting department');
            }
        }).catch(error => {
            alert('Error deleting department');
        });
    }
}

// Employee Management Functions
function viewEmployee(employeeId) {
    window.location.href = `/staff/view/${employeeId}`;
}

function deleteEmployee(employeeId) {
    if (confirm('Are you sure you want to delete this employee?')) {
        console.log('Delete employee:', employeeId);
        // Implementation for deleting employee
    }
}

// Custom Field Management
function showAddFieldModal() {
    const modal = document.getElementById('addFieldModal');
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

function toggleFieldOptions() {
    const fieldType = document.getElementById('field_type').value;
    const optionsGroup = document.getElementById('optionsGroup');
    
    if (fieldType === 'select') {
        optionsGroup.style.display = 'block';
        document.getElementById('field_options').required = true;
    } else {
        optionsGroup.style.display = 'none';
        document.getElementById('field_options').required = false;
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'info'}-circle"></i>
        <span>${message}</span>
        <button class="notification-close">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
    
    notification.querySelector('.notification-close').addEventListener('click', function() {
        notification.parentNode.removeChild(notification);
    });
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    }).format(new Date(date));
}

// Global functions
window.hrApp = {
    showNotification,
    formatCurrency,
    formatDate,
    deleteDepartment,
    deleteEmployee,
    viewEmployee,
    showAddFieldModal,
    closeModal,
    toggleFieldOptions,
    // NEW: Export biometric validation functions for reuse
    validateBiometricIdInForm,
    showFieldError,
    clearFieldError
};