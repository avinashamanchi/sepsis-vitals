import { useEffect, useRef } from 'react'
import { useStore } from '../stores/useStore'

const WS_RECONNECT_DELAY = 3000
const WS_MAX_RECONNECT_DELAY = 30000

function playAlertSound(level: string) {
  try {
    const ctx = new AudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.value = level === 'critical' ? 880 : 660
    gain.gain.value = 0.3
    osc.start()
    osc.stop(ctx.currentTime + (level === 'critical' ? 0.3 : 0.15))
  } catch {
    // AudioContext not available
  }
}

export function useWebSocket() {
  const setWsConnected = useStore((s) => s.setWsConnected)
  const setWsState = useStore((s) => s.setWsState)
  const addAlert = useStore((s) => s.addAlert)
  const updatePatientRisk = useStore((s) => s.updatePatientRisk)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelay = useRef(WS_RECONNECT_DELAY)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const hasConnectedOnce = useRef(false)

  useEffect(() => {
    // Don't connect on GitHub Pages (no backend) — stay at 'offline'
    if (window.location.hostname.includes('github.io')) {
      return
    }

    let unmounted = false

    function connect() {
      // Bail if the component was unmounted while a reconnect was pending
      if (unmounted) return

      // Build WebSocket URL from current location
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      let host: string
      try {
        host = import.meta.env.VITE_API_URL
          ? new URL(import.meta.env.VITE_API_URL).host
          : window.location.host
      } catch {
        setWsState('offline')
        return
      }
      let token: string | null = null
      try { token = localStorage.getItem('sv_token') } catch { /* private browsing */ }
      const params = token ? `?token=${token}` : ''
      const url = `${protocol}//${host}/ws/alerts${params}`

      // If we've connected before, this is a reconnect attempt
      if (hasConnectedOnce.current) {
        setWsState('reconnecting')
      }

      try {
        const ws = new WebSocket(url)
        wsRef.current = ws

        ws.onopen = () => {
          if (unmounted) { ws.close(); return }
          setWsConnected(true)
          setWsState('connected')
          hasConnectedOnce.current = true
          reconnectDelay.current = WS_RECONNECT_DELAY
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'sepsis_alert') {
              const riskLevel = data.risk_level ?? 'high'
              addAlert({
                id: `ws-${Date.now()}-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10)}`,
                patientId: data.patient_id ?? '',
                riskLevel,
                riskProbability: data.risk_probability ?? 0,
                message: data.recommendation ?? `Sepsis alert: ${riskLevel}`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
              // Play sound for high/critical alerts
              if (riskLevel === 'critical' || riskLevel === 'high') {
                playAlertSound(riskLevel)
              }
            }

            if (data.type === 'patient_update' || data.type === 'deterioration_alert' || data.type === 'recovery_alert') {
              updatePatientRisk(data.patient_id, {
                risk_probability: data.risk_probability ?? 0,
                risk_level: data.risk_level ?? 'low',
                last_prediction_time: Date.now() / 1000,
              })
            }

            if (data.type === 'deterioration_alert') {
              const riskLevel = data.risk_level ?? 'high'
              addAlert({
                id: `ws-det-${Date.now()}-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10)}`,
                patientId: data.patient_id ?? '',
                riskLevel,
                riskProbability: data.risk_probability ?? 0,
                message: `Deterioration: risk increased by ${((data.risk_delta ?? 0) * 100).toFixed(0)}% over ${(data.window_hours ?? 0).toFixed(1)}h`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
              playAlertSound(riskLevel)
            }

            if (data.type === 'recovery_alert') {
              addAlert({
                id: `ws-rec-${Date.now()}-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10)}`,
                patientId: data.patient_id ?? '',
                riskLevel: data.risk_level ?? 'low',
                riskProbability: data.risk_probability ?? 0,
                message: `Recovery: risk decreased by ${(Math.abs(data.risk_delta ?? 0) * 100).toFixed(0)}%`,
                timestamp: data.timestamp ?? new Date().toISOString(),
                dismissed: false,
              })
            }
          } catch {
            // ignore malformed messages
          }
        }

        ws.onclose = () => {
          if (unmounted) return
          setWsConnected(false)
          setWsState('reconnecting')
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
        if (!unmounted) {
          setWsConnected(false)
          setWsState('offline')
        }
      }
    }

    connect()

    return () => {
      unmounted = true
      clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect on intentional close
        wsRef.current.close()
      }
      setWsConnected(false)
      setWsState('offline')
    }
  }, [setWsConnected, setWsState, addAlert, updatePatientRisk])
}
