// Restore deep links forwarded by docs/404.html on GitHub Pages.
const forwardedPath = new URLSearchParams(window.location.search).get('p')
if (forwardedPath?.startsWith('/sepsis-vitals/')) {
  window.history.replaceState(null, '', forwardedPath)
}
