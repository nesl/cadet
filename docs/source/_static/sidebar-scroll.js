(function () {
  'use strict';
  var KEY = 'sidebarScrollTop';
  var nav = document.querySelector('.sidebar-scroll');
  if (!nav) return;

  function offsetFromNav(el, container) {
    var offset = 0;
    while (el && el !== container) {
      offset += el.offsetTop;
      el = el.offsetParent;
    }
    return offset;
  }

  // Restore saved scroll position.
  try {
    var saved = sessionStorage.getItem(KEY);
    if (saved !== null) nav.scrollTo({ top: parseInt(saved, 10), behavior: 'instant' });
  } catch (_) {}

  // Scroll the active link into view inside the sidebar, if needed.
  var active = nav.querySelector('.current-page') ||
               nav.querySelector('[aria-current="page"]') ||
               nav.querySelector('.current');
  if (active && nav.scrollHeight > nav.clientHeight) {
    var linkTop = offsetFromNav(active, nav);
    var linkHeight = active.offsetHeight;
    var navHeight = nav.clientHeight;
    var scrollTop = nav.scrollTop;
    if (linkTop < scrollTop || linkTop + linkHeight > scrollTop + navHeight) {
      nav.scrollTo({ top: linkTop - navHeight / 3, behavior: 'instant' });
    }
  }

  // Save on unload.
  window.addEventListener('pagehide', function () {
    try { sessionStorage.setItem(KEY, nav.scrollTop); } catch (_) {}
  });
})();
