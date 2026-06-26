import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import clsx from 'clsx'

interface TrendArrowProps {
  direction: 'improving' | 'stable' | 'worsening' | 'unknown'
  rateText?: string
  size?: 'sm' | 'md'
}

export function TrendArrow({ direction, rateText, size = 'sm' }: TrendArrowProps) {
  const iconSize = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4'
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm'

  if (direction === 'worsening') {
    return (
      <span className={clsx('inline-flex items-center gap-1 text-danger', textSize)}>
        <TrendingUp className={iconSize} />
        {rateText && <span>{rateText}</span>}
      </span>
    )
  }
  if (direction === 'improving') {
    return (
      <span className={clsx('inline-flex items-center gap-1 text-accent', textSize)}>
        <TrendingDown className={iconSize} />
        {rateText && <span>{rateText}</span>}
      </span>
    )
  }
  return (
    <span className={clsx('inline-flex items-center gap-1 text-text-muted', textSize)}>
      <Minus className={iconSize} />
      {rateText && <span>{rateText}</span>}
    </span>
  )
}
