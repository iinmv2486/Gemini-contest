function showTranslationPopup(translatedTexts) {
    let existingPopup = document.getElementById('translation-popup');
    if (existingPopup) {
        existingPopup.remove();
    }

    let popup = document.createElement('div');
    popup.id = 'translation-popup';
    popup.innerHTML = `
        <div class="popup-header">
            <div class="logo">
                <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="TransMate Logo">
                <div class="title">TransMate</div>
            </div>
            <div class="controls">
                <img id="settings-button" src="${chrome.runtime.getURL('icons/settings.svg')}" alt="Settings">
                <img id="close-popup" src="${chrome.runtime.getURL('icons/close.svg')}" alt="Close">
            </div>
        </div>
        <div class="translated-text">${translatedTexts}</div>
    `;

    document.body.appendChild(popup);

    popup.style.position = 'fixed';
    popup.style.top = '20px';
    popup.style.right = '20px';

    document.getElementById('close-popup').addEventListener('click', function() {
        popup.remove();
    });

    document.getElementById('settings-button').addEventListener('click', function() {
        chrome.runtime.sendMessage({ type: 'openOptionsPage' });
    });
    
}