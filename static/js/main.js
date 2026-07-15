/**
 * MindSight – Main Interactions & Animations
 * Handles scroll reveal, hover effects, and micro-interactions
 */

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initGlassHover();
  initSmoothScroll();
});

/**
 * Scroll Reveal Animation logic
 */
function initScrollReveal() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('reveal-visible');
        // Optional: stop observing once revealed
        // observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Select elements to reveal
  const revealElements = document.querySelectorAll(
    '.feature-card, .model-card, .risk-card, .section-header, .about-text, .about-image-wrap, .contact-container, .disclaimer-container'
  );

  revealElements.forEach(el => {
    el.classList.add('reveal-hidden');
    observer.observe(el);
  });
}

/**
 * Interactive glassmorphism hover effect
 */
function initGlassHover() {
  const cards = document.querySelectorAll('.glass-card, .feature-card, .model-card');

  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);
    });
  });
}

/**
 * Enhanced smooth scrolling for anchor links
 */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;

      const target = document.querySelector(targetId);
      if (target) {
        const headerOffset = 80;
        const elementPosition = target.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });
}
