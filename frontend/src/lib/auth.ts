import {
  signInWithPopup,
  GoogleAuthProvider,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  onAuthStateChanged,
  type User,
} from 'firebase/auth'
import { getFirebaseAuth } from './firebase'

const googleProvider = new GoogleAuthProvider()

function requireAuth() {
  const a = getFirebaseAuth()
  if (!a) throw new Error('Firebase is not configured')
  return a
}

export async function signInWithGoogle() {
  const result = await signInWithPopup(requireAuth(), googleProvider)
  return result.user
}

export async function signInWithEmail(email: string, password: string) {
  const result = await signInWithEmailAndPassword(requireAuth(), email, password)
  return result.user
}

export async function signUpWithEmail(email: string, password: string) {
  const result = await createUserWithEmailAndPassword(requireAuth(), email, password)
  return result.user
}

export async function signOutUser() {
  const a = getFirebaseAuth()
  if (a) await signOut(a)
}

export async function resetPassword(email: string) {
  await sendPasswordResetEmail(requireAuth(), email)
}

export function onAuthChange(callback: (user: User | null) => void) {
  const a = getFirebaseAuth()
  if (!a) {
    // No Firebase configured (demo mode) — invoke with null immediately
    callback(null)
    return () => {}
  }
  return onAuthStateChanged(a, callback)
}
