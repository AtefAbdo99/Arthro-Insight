document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    const currentTheme = localStorage.getItem('theme') || 'light';
    body.classList.toggle('dark-mode', currentTheme === 'dark');

    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-mode');
        const theme = body.classList.contains('dark-mode') ? 'dark' : 'light';
        localStorage.setItem('theme', theme);
    });

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    const form = document.querySelector('.prediction-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            form.querySelectorAll('select').forEach(select => {
                if (!select.value) {
                    isValid = false;
                    select.classList.add('error');
                } else {
                    select.classList.remove('error');
                }
            });

            if (!isValid) {
                e.preventDefault();
                alert('Please fill out all fields before submitting.');
            }
        });
    }

    const resultCards = document.querySelectorAll('.result-card');
    resultCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });

    const bestModelCard = document.querySelector('.best-model');
    if (bestModelCard) {
        bestModelCard.classList.add('circle-animation');
    }

    const socialLinks = document.querySelectorAll('.social-links img');
    socialLinks.forEach(link => {
        link.addEventListener('mouseover', () => {
            link.classList.add('hover');
        });
        link.addEventListener('mouseout', () => {
            link.classList.remove('hover');
        });
    });
});
