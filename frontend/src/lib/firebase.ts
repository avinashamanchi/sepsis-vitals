import { initializeApp, type FirebaseApp } from 'firebase/app'
import { getAuth, type Auth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? 'AIzaSyAG4yRPg8lIO9K191Rdc_gAZ5wMaS9PKco',
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? 'sepsis-a16be.firebaseapp.com',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? 'sepsis-a16be',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? 'sepsis-a16be.firebasestorage.app',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? '11649683584',
  appId: import.meta.env.VITE_FIREBASE_APP_ID ?? '1:11649683584:web:929e7eb59d52383c889665',
}

/** Firebase is initialized lazily so the app doesn't crash in demo mode
 *  (GitHub Pages) where no Firebase env vars are set. */
let _app: FirebaseApp | null = null
let _auth: Auth | null = null

export function getFirebaseAuth(): Auth | null {
  if (!firebaseConfig.apiKey) return null
  if (!_app) _app = initializeApp(firebaseConfig)
  if (!_auth) _auth = getAuth(_app)
  return _auth
}
