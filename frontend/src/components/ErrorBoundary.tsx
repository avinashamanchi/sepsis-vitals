import { Component, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Global error boundary that catches any unhandled render errors
 * and shows a recovery UI instead of a blank white screen.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary] Uncaught render error:', error, info.componentStack)

    // Auto-recover from stale chunk errors (e.g. after a new deploy
    // when the service worker still caches old asset filenames)
    if (this.isChunkLoadError(error)) {
      navigator.serviceWorker?.getRegistrations().then((regs) =>
        Promise.all(regs.map((r) => r.unregister()))
      ).finally(() => window.location.reload())
    }
  }

  private isChunkLoadError(error: Error): boolean {
    const msg = error.message ?? ''
    return msg.includes('dynamically imported module') ||
           msg.includes('Failed to fetch') ||
           msg.includes('Loading chunk') ||
           msg.includes('Loading CSS chunk')
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background font-mono flex items-center justify-center p-4">
          <div className="w-full max-w-md text-center">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-danger/10 border border-danger/20 mb-4">
              <AlertTriangle className="w-7 h-7 text-danger" />
            </div>
            <h1 className="font-heading text-xl font-bold text-text-primary mb-2">
              Something went wrong
            </h1>
            <p className="text-sm text-text-secondary mb-6">
              An unexpected error occurred. Your data is safe.
            </p>
            {this.state.error && (
              <pre className="text-xs text-text-muted bg-surface border border-border rounded-lg p-3 mb-6 text-left overflow-auto max-h-32">
                {this.state.error.message}
              </pre>
            )}
            <div className="flex gap-3 justify-center">
              <button
                onClick={this.handleRetry}
                className="flex items-center gap-2 px-4 py-2.5 text-sm font-medium bg-accent/10 text-accent border border-accent/30 rounded-lg hover:bg-accent/20 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Try Again
              </button>
              <button
                onClick={this.handleReload}
                className="px-4 py-2.5 text-sm font-medium text-text-secondary border border-border rounded-lg hover:bg-elevated transition-colors"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
