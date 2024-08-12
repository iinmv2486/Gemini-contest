chrome.runtime.onMessage.addListener((message, sender, sendResponse) => { // 옵션버튼 눌렸는지 확인
  if (message.type === 'openOptionsPage') {
      chrome.runtime.openOptionsPage();
  }
});

// 확장 프로그램이 설치되거나 업데이트될 때 실행
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "translateWithTransMate",
    title: "Translate with TransMate",
    contexts: ["page", "selection"]
  }, () => {
    if (chrome.runtime.lastError) {
      console.error("Error creating context menu:", chrome.runtime.lastError);
    } else {
      console.log("Context menu created successfully");
    }
  });
});

// 컨텍스트 메뉴 클릭 이벤트 처리
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "translateWithTransMate") {
    chrome.tabs.sendMessage(tab.id, { type: 'TranslatePage' });
  }
});

// ... (기존 코드는 그대로 유지)

// 메시지 리스너
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'originalText' || message.type === 'TranslateSelectedText') {
    console.log("message!")
    console.log(message)
    handleTranslation(message, sender.tab.id, message.data.randomKey);
  } else if (message.type === 'LanguageChanged') {
    // 언어 변경 처리 (기존 코드 유지)
    chrome.storage.sync.set({ language: message.language }, () => {
      console.log('Language updated to:', message.language);
    });}
    else if (message.type === 'SettingsChanged') {
      // 모든 탭에 설정 변경 메시지 전송
      chrome.tabs.query({}, (tabs) => {
        tabs.forEach((tab) => {
          chrome.tabs.sendMessage(tab.id, {
            type: 'SettingsChanged',
            settings: message.settings
          });
        });
      });
      // options 페이지에도 설정 변경 메시지 전송
    chrome.runtime.sendMessage({
      type: 'SettingsChanged',
      settings: message.settings
    });


    }
  
});

// 번역 처리 함수
function handleTranslation(message, tabId, randomKey) {
  console.log("번역 처리 시작");
  const texts = message.data.originalText;
  console.log("번역할 텍스트:", texts);

  chrome.storage.sync.get('language', (data) => {
    const lang = data.language || 'ko'; // 기본 언어를 한국어로 설정
    translateTexts(texts, lang)
      .then((translatedTexts) => {
        if (message.type === 'originalText'){
        chrome.tabs.sendMessage(tabId, {
          type: 'TranslatedText',
          data: {strs:translatedTexts, randomKey:randomKey},
        })}
        else if (message.type === 'TranslateSelectedText'){
          chrome.tabs.sendMessage(tabId, {
            type: 'TranslatedSelectedText',
            data: {strs:translatedTexts},
          })};
      })
      .catch((error) => {
        console.error('Translation error:', error);
        const failedTranslations = texts.map(() => 'translation failed');
        chrome.tabs.sendMessage(tabId, {
          type: 'TranslatedText',
          data: failedTranslations,
        });
      });
  });
}

// 기존의 translateTexts 및 callTranslationAPI 함수는 그대로 유지

// 번역 API 호출 함수
async function translateTexts(texts, lang) {
  try {
    console.log(1);
    console.log(JSON.stringify({ texts: texts, language: lang }));
    const translatedTexts = await callTranslationAPI(texts, lang);
    return translatedTexts;
  } catch (error) {
    console.error('Translation API call failed:', error);
    return texts.map(() => 'api에서 번역 실패');
  }
}

// 번역 API 호출 함수
async function callTranslationAPI(texts, lang) {
  console.log(2);
  console.log(JSON.stringify({ strs: texts, language: lang }));
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 30초 타임아웃

  const response = await fetch('http://158.247.199.223:3001/translate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ strs: texts, language: lang }), // body에 원문 text 넣어서 json 형태로 전달
    signal: controller.signal,
  });

  clearTimeout(timeoutId);

  if (!response.ok) {
    throw new Error('Translation API failed');
  }
  const data = await response.json();
  console.log('request에 담은 내용');
  console.log(JSON.stringify({ strs: texts, language: lang }))
  console.log("response.body");
  console.log(data)
  //console.log(data);
  return data;
}


// background.js에 추가

self.addEventListener('fetch', (event) => {
  const preloadPromise = event.preloadResponse;
  event.waitUntil(
      preloadPromise.then(response => {
          if (response) {
              return response;
          }
          return fetch(event.request);
      }).catch(error => {
          console.error('Preload response failed:', error);
          return fetch(event.request);
      })
  );
});
