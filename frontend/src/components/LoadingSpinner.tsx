import clsx from 'clsx'
import { useTranslation } from 'react-i18next'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  label?: string
  className?: string
}

export function LoadingSpinner({ size = 'md', label, className }: LoadingSpinnerProps) {
  const { t } = useTranslation()

  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-3',
  }

  return (
    <div className={clsx('flex flex-col items-center justify-center gap-3', className)}>
      <div
        className={clsx(
          'rounded-full border-text-muted/30 border-t-accent animate-spin',
          sizeClasses[size],
        )}
        role="status"
        aria-label={label ?? t('common.loading')}
      />
      {label && <p className="text-sm text-text-muted">{label}</p>}
    </div>
  )
}
