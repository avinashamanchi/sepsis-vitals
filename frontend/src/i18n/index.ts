import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import en from '../locales/en.json'
import fr from '../locales/fr.json'
import pt from '../locales/pt.json'
import ar from '../locales/ar.json'
import sw from '../locales/sw.json'
import am from '../locales/am.json'

export const LANGUAGES = [
  { code: 'en', name: 'English', dir: 'ltr' },
  { code: 'fr', name: 'Français', dir: 'ltr' },
  { code: 'pt', name: 'Português', dir: 'ltr' },
  { code: 'ar', name: 'العربية', dir: 'rtl' },
  { code: 'sw', name: 'Kiswahili', dir: 'ltr' },
  { code: 'am', name: 'አማርኛ', dir: 'ltr' },
] as const

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      fr: { translation: fr },
      pt: { translation: pt },
      ar: { translation: ar },
      sw: { translation: sw },
      am: { translation: am },
    },
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
    },
  })

export default i18n
