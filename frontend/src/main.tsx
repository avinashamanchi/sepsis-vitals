import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import './i18n'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { initOutbox, subscribe as subscribeOutbox } from './lib/outbox'
import { useStore } from './stores/useStore'

// Catch unhandled promise rejections globally so they never crash the page
window.addEventListener('unhandledrejection', (event) => {
  console.error('[Global] Unhandled promise rejection:', event.reason)
  event.preventDefault()
})

// Initialize offline-first vitals outbox
initOutbox()
subscribeOutbox((count) => useStore.getState().setOutboxPending(count))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter basename="/sepsis-vitals">
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
)
