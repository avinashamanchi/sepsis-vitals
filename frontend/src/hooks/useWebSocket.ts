import { useEffect, useRef } from 'react'
import { useStore } from '../stores/useStore'

const WS_RECONNECT_DELAY = 3000
const WS_MAX_RECONNECT_DELAY = 30000

export function useWebSocket() {
  const setWsConnected = useStore((s) => s.setWsConnected)
  const addAlert = useStore((s) => s.addAlert)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelay = useRef(WS_RECONNECT_DELAY)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    function connect() {
      // Build WebSocket URL from current location
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = import.meta.env.VITE_API_URL
        ? new URL(import.meta.env.VITE_API_URL).host
        : window.location.host
      const token = localStorage.getItem('sv_token')
      const params = token ? `?token=${token}` : ''
      const url = `${protocol}//${host}/ws/alerts${params}`

      // Don't connect if we're on GitHub Pages (no backend)
      if (window.location.hostname.includes('github.io')) {
        return
      }

      try {
        const ws = new WebSocket(url)
        wsRef.current = ws

        ws.onopen = () => {
          setWsConnected(true)
          reconnectDelay.current = WS_RECONNECT_DELAY
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'sepsis_alert') {
              addAlert({
                id: `ws-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
                patientId: data.patient_id ?? '',
                riskLevel: data.risk_level ?? 'high',
                riskProbability: data.risk_probability ?? 0,
                message: data.recommendation ?? `Sepsis alert: ${data.risk_level}`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
            }
          } catch {
            // ignore malformed messages
          }
        }

        ws.onclose = () => {
          setWsConnected(false)
          wsRef.current = null
          // Exponential backoff reconnect
          reconnectTimer.current = setTimeout(() => {
            reconnectDelay.current = Math.min(
              reconnectDelay.current * 1.5,
              WS_MAX_RECONNECT_DELAY,
            )
            connect()
          }, reconnectDelay.current)
        }

        ws.onerror = () => {
          ws.close()
        }
      } catch {
        // WebSocket constructor can throw if URL is invalid
        setWsConnected(false)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect on intentional close
        wsRef.current.close()
      }
      setWsConnected(false)
    }
  }, [setWsConnected, addAlert])
}
