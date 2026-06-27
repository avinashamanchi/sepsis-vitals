import { useHotkeys } from 'react-hotkeys-hook'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../stores/useStore'

export function KeyboardShortcuts() {
  const navigate = useNavigate()
  const alerts = useStore((s) => s.alerts)
  const acknowledgeAlert = useStore((s) => s.acknowledgeAlert)

  // Cmd/Ctrl + K → Navigate to patients and focus search
  useHotkeys('mod+k', (e) => {
    e.preventDefault()
    navigate('/patients')
    // Focus the search input after navigation
    setTimeout(() => {
      const searchInput = document.querySelector<HTMLInputElement>('[data-search-patients]')
      searchInput?.focus()
    }, 100)
  }, { enableOnFormTags: false })

  // Alt + N → Navigate to predict (new vitals)
  useHotkeys('alt+n', (e) => {
    e.preventDefault()
    navigate('/predict')
  }, { enableOnFormTags: false })

  // Escape → Acknowledge top-most active alert
  useHotkeys('escape', () => {
    const topAlert = alerts.find((a) => !a.dismissed && !a.acknowledged)
    if (topAlert) {
      acknowledgeAlert(topAlert.id)
    }
  }, { enableOnFormTags: false })

  return null
}
