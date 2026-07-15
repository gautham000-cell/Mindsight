// ================================================
//  MindSight – Screening Form JS
// ================================================

let currentStep = 1;
const totalSteps = 4;

document.addEventListener('DOMContentLoaded', () => {
    initToggles();
    initSliders(); // Added to fix dead sliders
    updateProgress();

    // Form submit
    document.getElementById('screening-form-el').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitForm();
    });
});

// ---- Slider value updates ----
function initSliders() {
    const sliders = [
        'sleep_hours', 'physical_activity', 'social_support', 'work_life_balance',
        'screen_time', 'mindfulness_practice', 'stress_level', 'self_esteem',
        'coping_skills', 'phq9_score', 'gad7_score', 'pss_score'
    ];

    sliders.forEach(id => {
        const slider = document.getElementById(id);
        const valDisp = document.getElementById(id + '_val');
        if (slider && valDisp) {
            const updateVal = () => {
                let suffix = '';
                if (id === 'sleep_hours' || id === 'screen_time') suffix = ' hrs';
                valDisp.textContent = slider.value + suffix;
            };
            slider.addEventListener('input', updateVal);
            updateVal(); // Initial set
        }
    });
}

// ---- Multi-step navigation ----
function nextStep() {
    if (!validateStep(currentStep)) return;
    if (currentStep < totalSteps) {
        showStep(currentStep + 1);
    }
}

function prevStep() {
    if (currentStep > 1) {
        showStep(currentStep - 1);
    }
}

function showStep(step) {
    document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
    document.querySelector(`[data-step="${step}"]`).classList.add('active');
    currentStep = step;
    updateProgress();
    updateNavButtons();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateProgress() {
    const pct = (currentStep / totalSteps) * 100;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-text').textContent = `Step ${currentStep} of ${totalSteps}`;
}

function updateNavButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const submitBtn = document.getElementById('submit-btn');

    prevBtn.style.display = currentStep > 1 ? 'inline-flex' : 'none';
    nextBtn.style.display = currentStep < totalSteps ? 'inline-flex' : 'none';
    submitBtn.style.display = currentStep === totalSteps ? 'inline-flex' : 'none';
}

// ---- Validation ----
function validateStep(step) {
    if (step === 1) {
        const name = document.getElementById('name').value.trim();
        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const occ = document.getElementById('occupation').value;
        if (!name || !age || !gender || !occ) {
            showToast('Please fill in all required fields.');
            return false;
        }
        if (parseInt(age) < 18 || parseInt(age) > 100) {
            showToast('Age must be between 18 and 100.');
            return false;
        }
    }
    return true;
}

// ---- Toggle buttons for binary questions ----
function initToggles() {
    document.querySelectorAll('.toggle-group').forEach(group => {
        const name = group.dataset.name;
        const hiddenInput = document.getElementById(name);
        group.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                if (hiddenInput) hiddenInput.value = btn.dataset.value;
            });
        });
    });
}

// ---- Collect form data ----
function collectFormData() {
    const getValue = (id) => {
        const el = document.getElementById(id);
        return el ? el.value : '';
    };
    return {
        name: getValue('name'),
        age: getValue('age'),
        gender: getValue('gender'),
        occupation: getValue('occupation'),
        sleep_hours: getValue('sleep_hours'),
        physical_activity: getValue('physical_activity'),
        social_support: getValue('social_support'),
        work_life_balance: getValue('work_life_balance'),
        screen_time: getValue('screen_time'),
        mindfulness_practice: getValue('mindfulness_practice'),
        stress_level: getValue('stress_level'),
        self_esteem: getValue('self_esteem'),
        coping_skills: getValue('coping_skills'),
        family_history: getValue('family_history'),
        chronic_illness: getValue('chronic_illness'),
        substance_use: getValue('substance_use'),
        previous_treatment: getValue('previous_treatment'),
        phq9_score: getValue('phq9_score'),
        gad7_score: getValue('gad7_score'),
        pss_score: getValue('pss_score'),
    };
}

// ---- Submit & Get Prediction ----
async function submitForm() {
    const submitBtn = document.getElementById('submit-btn');
    const submitText = document.getElementById('submit-text');
    const spinner = document.getElementById('spinner');

    submitBtn.disabled = true;
    submitText.textContent = 'Analyzing...';
    spinner.style.display = 'block';

    try {
        const data = collectFormData();
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        if (result.success) {
            showResults(result);
        } else {
            showToast('Prediction failed: ' + result.error);
        }
    } catch (err) {
        showToast('Connection error. Please try again.');
        console.error(err);
    } finally {
        submitBtn.disabled = false;
        submitText.textContent = 'Get My Assessment';
        spinner.style.display = 'none';
    }
}

// ---- Render Results ----
function showResults(result) {
    document.getElementById('screening-form').style.display = 'none';
    const panel = document.getElementById('results-panel');
    panel.style.display = 'block';
    panel.style.animation = 'fadeIn 0.5s ease';

    // Name
    document.getElementById('result-name').textContent = result.name || 'You';

    // Risk Badge
    const badge = document.getElementById('risk-badge');
    const risk = result.risk_level;
    badge.textContent = risk + ' Risk';
    badge.className = 'risk-badge-large risk-badge-' + risk.toLowerCase();

    // Risk score
    const score = parseFloat(result.risk_score);
    document.getElementById('risk-score-num').textContent = score.toFixed(1);

    // Confidence score
    document.getElementById('confidence-score-num').textContent = (result.confidence_score || 0).toFixed(1) + '%';

    // Score bar color
    const fill = document.getElementById('score-fill');
    const pct = (score / 10) * 100;
    let color = '#22c55e';
    if (risk === 'Moderate') color = '#f59e0b';
    if (risk === 'High') color = '#ef4444';
    setTimeout(() => {
        fill.style.width = pct + '%';
        fill.style.background = color;
    }, 200);

    // Result icon
    const icons = { Low: '😊', Moderate: '😟', High: '🚨' };
    document.getElementById('result-icon').textContent = icons[risk] || '🧠';

    // Probabilities
    const probBars = document.getElementById('prob-bars');
    probBars.innerHTML = '';
    const probColors = { Low: '#22c55e', Moderate: '#f59e0b', High: '#ef4444' };
    Object.entries(result.probabilities || {}).forEach(([label, pct]) => {
        const div = document.createElement('div');
        div.className = 'prob-row';
        div.innerHTML = `
      <span class="prob-label">${label}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:0%; background:${probColors[label] || '#6366f1'}" data-target="${pct}"></div>
      </div>
      <span class="prob-val" style="color:${probColors[label]}">${pct}%</span>
    `;
        probBars.appendChild(div);
    });
    setTimeout(() => {
        probBars.querySelectorAll('.prob-fill').forEach(bar => {
            bar.style.width = bar.dataset.target + '%';
        });
    }, 300);

    // Model name
    document.getElementById('model-used-name').textContent = result.model_used || 'ML Model';

    // Recommendations
    const recGrid = document.getElementById('rec-grid');
    recGrid.innerHTML = '';
    (result.recommendations || []).forEach((rec, i) => {
        const card = document.createElement('div');
        card.className = 'rec-card';
        card.style.animationDelay = (i * 0.08) + 's';
        card.innerHTML = `
      <div class="rec-icon">${rec.icon || '💡'}</div>
      <div>
        <div class="rec-category">${rec.category || ''}</div>
        <div class="rec-text">${rec.advice || ''}</div>
      </div>
    `;
        recGrid.appendChild(card);
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---- Reset ----
function resetForm() {
    document.getElementById('results-panel').style.display = 'none';
    document.getElementById('screening-form').style.display = 'block';
    showStep(1);
    document.getElementById('screening-form-el').reset();
    // Reset toggles
    document.querySelectorAll('.toggle-group').forEach(group => {
        const btns = group.querySelectorAll('.toggle-btn');
        btns.forEach(b => b.classList.remove('active'));
        if (btns.length) btns[0].classList.add('active');
        const name = group.dataset.name;
        const hidden = document.getElementById(name);
        if (hidden) hidden.value = '0';
    });
    // Re-initialize sliders
    initSliders();
}

// ---- Toast notification ----
function showToast(msg) {
    const existing = document.getElementById('toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.textContent = msg;
    toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#ef4444; color:white; padding:14px 22px;
    border-radius:10px; font-size:0.9rem; font-weight:600;
    box-shadow:0 8px 24px rgba(239,68,68,0.4);
    animation:fadeIn 0.3s ease;
    max-width:320px;
  `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}
