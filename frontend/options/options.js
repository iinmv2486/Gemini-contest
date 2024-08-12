document.addEventListener('DOMContentLoaded', () => {
  // 저장된 언어 설정 불러오기
  // 슬라이더 설정 불러오기
  chrome.storage.sync.get(['tooltipEnabled'], (data) => {
    if (data.tooltipEnabled !== undefined) {
      document.getElementById('tooltipToggle').checked = data.tooltipEnabled;
    }
    else {
        // 최초 설치 시 기본값을 ON으로 설정
        document.getElementById('tooltipToggle').checked = true;
        chrome.storage.sync.set({ tooltipEnabled: true });
    }
  });
  // 설정 변경 메시지 리스너 추가
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'SettingsChanged') {
      if (message.settings.tooltipEnabled !== undefined) {
        document.getElementById('tooltipToggle').checked = message.settings.tooltipEnabled;
      }
    }
  });
  
  const logoImage = document.getElementById('logoImage');
  logoImage.src = chrome.runtime.getURL('icons/icon128.png');
  chrome.storage.sync.get(['language', 'tooltipEnabled'], (data) => {
      if (data.language) {
          document.getElementById('languageSelect').value = data.language;
      }
      if (data.tooltipEnabled !== undefined) {
          document.getElementById('tooltipToggle').checked = data.tooltipEnabled;
      }
  });
});

document.getElementById('saveSettings').addEventListener('click', () => {
    const language = document.getElementById('languageSelect').value;
    const tooltipEnabled = document.getElementById('tooltipToggle').checked;
  
    chrome.storage.sync.set({ 
        language: language,
        tooltipEnabled: tooltipEnabled
    }, () => {
        console.log('Settings saved:', { language, tooltipEnabled });
        // 설정이 저장된 후 백그라운드 스크립트에 알림
        chrome.runtime.sendMessage({ 
            type: 'SettingsChanged', 
            settings: { language, tooltipEnabled } 
        });
    });
  });