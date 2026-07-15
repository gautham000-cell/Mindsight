// ================================================
//  MindSight – Dashboard JS
// ================================================

document.addEventListener('DOMContentLoaded', () => {
    animateKPIs();
    animateGauges();
});

function animateKPIs() {
    document.querySelectorAll('.kpi-number[data-target]').forEach(el => {
        const target = parseFloat(el.dataset.target);
        const suffix = el.dataset.suffix || '';
        const isDecimal = target % 1 !== 0;
        const duration = 1500;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const val = target * eased;
            el.textContent = (isDecimal ? val.toFixed(1) : Math.floor(val)) + suffix;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    });
}

function animateGauges() {
    // Gauge bars start at 0 then animate to their target width
    document.querySelectorAll('.gauge-fill').forEach(el => {
        const w = el.style.width;
        el.style.width = '0%';
        setTimeout(() => { el.style.width = w; }, 200);
    });
}
