document.getElementById('saveSettings').addEventListener('click', () => {
  const language = document.getElementById('languageSelect').value;
  chrome.storage.sync.set({ language: language }, () => {
    console.log('Language setting saved:', language);
    // 설정이 저장된 후 백그라운드 스크립트에 알림
    chrome.runtime.sendMessage({ type: 'LanguageChanged', language: language });
  });
});
