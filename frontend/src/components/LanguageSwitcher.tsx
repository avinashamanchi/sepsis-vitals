import { useTranslation } from 'react-i18next'
import { LANGUAGES } from '../i18n'
import { Globe } from 'lucide-react'

export function LanguageSwitcher() {
  const { i18n } = useTranslation()

  return (
    <div className="relative inline-flex items-center gap-1.5">
      <Globe className="w-4 h-4 text-text-muted" aria-hidden="true" />
      <select
        value={i18n.language}
        onChange={(e) => i18n.changeLanguage(e.target.value)}
        className="bg-surface border border-border rounded px-2 py-1 text-xs text-text-primary cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent appearance-none pr-5"
        aria-label="Select language"
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  )
}
