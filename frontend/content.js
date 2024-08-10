
if (chrome.sidePanel) {//사이드 패널 지원하면 사이드 패널에 로고 띄워놓기
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
}
///// 부분번역
loadPopupCSS();
// 텍스트 드래그 시 이벤트 리스너 추가
document.addEventListener('mouseup', function() {
  const selectedText = window.getSelection().toString().trim();
  if (selectedText) {
    chrome.runtime.sendMessage({
      type: 'TranslateSelectedText',
      data: { originalText: [selectedText] }  // 배열로 변경
  });
  }
});
/////////////////////////////////////////
//전체번역
const bannedTagNames = [
  "SCRIPT",
  "SVG",
  "STYLE",
  "NOSCRIPT",
  "IFRAME",
  "OBJECT",
];

// 특정 요소를 건너뛸 수 있는지 확인하는 함수
const canSkip = (el) => {
  return (
    el.getAttribute?.("translate") === "no" || // 번역 제외 속성
    el.classList?.contains("notranslate") || // 번역 제외 클래스
    bannedTagNames.includes(el.tagName) || // 금지된 태그
    isInShadowDOM(el) // Shadow DOM에 포함된 요소
  );
};

const checkAllInline = (el) => {
  for (let i of el.childNodes) {
    if (canSkip(el)) {
      continue;
    }
    if (
      i.nodeType == Node.ELEMENT_NODE &&
      window.getComputedStyle(i).display !== "inline"
    ) {
      return false;
    }
  }
  return true;
};

// 요소가 Shadow DOM에 포함되어 있는지 확인하는 함수
const isInShadowDOM = (el) => {
  while (el) {
    if (el instanceof ShadowRoot) {
      return true;
    }
    el = el.parentNode;
  }
  return false;
};

// 모든 텍스트 노드를 수집하는 함수
const dfs = (el) => {
  if (canSkip(el)) {
    // 건너뛸 요소인지 확인
    return [];
  }

  let result = [];
  for (let i of el.childNodes) {
    if (canSkip(i)) {
      // 건너뛸 자식 요소인지 확인
      continue;
    }

    if (i.nodeType === Node.TEXT_NODE && i.textContent.trim()) {
      // 텍스트 노드인 경우
      result.push({ element: i, content: i.textContent.trim() });
    } else if (i.nodeType === Node.ELEMENT_NODE) {
      if (i.tagName === "BR") {
        // 줄바꿈 요소인 경우
        result.push({ element: i, content: "" });
        continue;
      }

      if (checkAllInline(i)) {
        // 인라인 요소인 경우 (초보자들을 위한 <span>Bronze</span> 같은 요소)
        let tmp = dfs(i);
        if (!tmp.length) continue;
        let div = document.createElement("div");
        for (let i of tmp) {
          if (i.element.nodeType === Node.TEXT_NODE) {
            div.appendChild(document.createTextNode(i.content));
          } else {
            let child = document.createElement(i.element.tagName);
            child.innerText = i.content;
            div.appendChild(child);
          }
        }
        result.push({ element: i, content: div.innerHTML });
      } else {
        // 요소 노드인 경우
        result.push(...dfs(i));
      }
    }
  }

  return result;
};

// 웹페이지의 모든 텍스트 요소를 가져오는 함수
function getAllTextNodes() {
  return dfs(document.body);
}

// 추출한 외국어 텍스트를 백그라운드 스크립트로 전송하는 함수
function sendForeignTextToBackground(textNodes,randomKey) {
  const textContents = textNodes.map((node, index) => ({
    index: index,
    content: node.content,
  }));
  console.log("추출된 텍스트");
  console.log(JSON.stringify({ textContents: textContents }));
  console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  console.log(textContents.map(item => item.content))
  chrome.runtime.sendMessage({
    type: "originalText",
    data: {
      originalText: textContents.map((item) => item.content),randomKey: randomKey // 여기서는 원래의 텍스트 내용만 전송
    },
  });
}

// 번역된 텍스트를 웹페이지에 적용하는 함수
function applyDfs(node, translatedNode) {
  let oriChilds = node.childNodes;
  let transChilds = translatedNode.childNodes;

  for (
    let oriIdx = 0, transIdx = 0;
    oriIdx < oriChilds.length;
    oriIdx++, transIdx++
  ) {
    if (
      oriChilds[oriIdx].textContent.trim() === "" &&
      oriChilds[oriIdx].tagName !== "BR"
    ) {
      transIdx--;
      continue;
    }

    if (transIdx >= transChilds.length) {
      if (oriChilds[oriIdx].nodeType === Node.TEXT_NODE) {
        oriChilds[oriIdx].textContent = "";
      } else {
        console.log(oriChilds[oriIdx]);
      }
      continue;
    }

    if (oriChilds[oriIdx].nodeType === Node.TEXT_NODE) {
      oriChilds[oriIdx].textContent = transChilds[transIdx].textContent;
    } else {
      applyDfs(oriChilds[oriIdx], transChilds[transIdx]);
    }
  }
}

function applyTranslatedText(textNodes, translatedTexts) {
  console.log("번역된 텍스트");
  console.log(translatedTexts);

  for (let i = 0; i < textNodes.length; i++) {
    if (textNodes[i].element.nodeType === Node.TEXT_NODE) {
      textNodes[i].element.textContent = translatedTexts[i];
    } else {
      let div = document.createElement("div");
      div.innerHTML = translatedTexts[i];
      applyDfs(textNodes[i].element, div);
    }
  }
}
let cache = {} 
let randomKey = Math.random().toString(36).substring(2, 12)
// 메시지 리스너 추가
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "TranslatePage") {
    let textNodes = getAllTextNodes();
    cache[randomKey] = textNodes;
    console.log(cache[randomKey]);
    sendForeignTextToBackground(textNodes,randomKey);
  } else if (message.type === "TranslatedText") {
    console.log("randonKey");
    console.log(randomKey);
    let textNodes = cache[message.data.randomKey]
    const translatedTexts = message.data.strs;
    console.log('translatedTexts!!!');
    console.log(translatedTexts);
    console.log(translatedTexts.strs.length)
    console.log("textNodes");
    console.log(textNodes);
    console.log(textNodes.length); 
    if (textNodes.length !== translatedTexts.strs.length) {
      console.error("번역된 텍스트의 수가 일치하지 않습니다.");
      return;
    }
    applyTranslatedText(textNodes, translatedTexts);
    console.log("applyTranslatedText 작동")
  }
  else if (message.type === 'TranslatedSelectedText') {
    const translatedTexts = message.data.strs[0];
    showTranslationPopup(translatedTexts);
  }
});

//