import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import './i18n'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { routerBasename } from './lib/basePath'

// Catch unhandled promise rejections globally so they never crash the page
window.addEventListener('unhandledrejection', (event) => {
  console.error('[Global] Unhandled promise rejection:', event.reason)
  event.preventDefault()
})

// A previous release cached API responses. Purge that cache because it could
// contain sensitive clinical data; only static application assets are cached.
if ('caches' in window) {
  void window.caches.delete('api-cache')
}
// Remove the discontinued offline-write database. It stored clinical payloads
// unencrypted and is not appropriate for this research release.
if ('indexedDB' in window) {
  window.indexedDB.deleteDatabase('sepsis-outbox')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter basename={routerBasename(import.meta.env.BASE_URL)}>
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
)
