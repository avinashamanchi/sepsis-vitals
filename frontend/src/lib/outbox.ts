/**
 * lib/outbox.ts
 * -------------
 * Offline-first write queue for a low-connectivity ward.
 *
 * The connection indicator was only ever a coloured dot — this makes "offline"
 * actually mean something. When a write (record vitals, complete a bundle task)
 * fails because the network is down, it is durably queued in IndexedDB and
 * replayed automatically when connectivity returns. Every queued item carries a
 * client-generated idempotency key so a replay can never double-write on the
 * backend (pair with an `Idempotency-Key` check on the relevant POST routes).
 *
 * Design notes:
 *  - IndexedDB (not localStorage) so payloads survive reloads and aren't capped
 *    at ~5 MB; a nurse can accumulate a shift's worth of readings offline.
 *  - FIFO replay preserves clinical ordering (cultures before antibiotics).
 *  - Exponential backoff with a cap; permanently-rejected items (4xx that
 *    aren't 408/429) are moved to a dead-letter store for review rather than
 *    blocking the queue forever.
 */

export interface OutboxItem {
  id: string
  idempotencyKey: string
  method: string
  path: string
  body: unknown
  createdAt: number
  attempts: number
  lastError?: string
}

const DB_NAME = 'sepsis-outbox'
const DB_VERSION = 1
const STORE = 'queue'
const DEAD = 'dead'

const MAX_ATTEMPTS = 8
const BASE_BACKOFF_MS = 2000
const MAX_BACKOFF_MS = 5 * 60 * 1000

type Listener = (pending: number) => void

let dbPromise: Promise<IDBDatabase> | null = null
let flushing = false
const listeners = new Set<Listener>()

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' })
      }
      if (!db.objectStoreNames.contains(DEAD)) {
        db.createObjectStore(DEAD, { keyPath: 'id' })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('IndexedDB open failed'))
  })
  return dbPromise
}

function tx(db: IDBDatabase, store: string, mode: IDBTransactionMode) {
  return db.transaction(store, mode).objectStore(store)
}

function uuid(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

async function allItems(): Promise<OutboxItem[]> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const req = tx(db, STORE, 'readonly').getAll()
    req.onsuccess = () =>
      resolve((req.result as OutboxItem[]).sort((a, b) => a.createdAt - b.createdAt))
    req.onerror = () => reject(req.error)
  })
}

async function put(store: string, item: OutboxItem): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const req = tx(db, store, 'readwrite').put(item)
    req.onsuccess = () => resolve()
    req.onerror = () => reject(req.error)
  })
}

async function remove(store: string, id: string): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const req = tx(db, store, 'readwrite').delete(id)
    req.onsuccess = () => resolve()
    req.onerror = () => reject(req.error)
  })
}

async function notify(): Promise<void> {
  const items = await allItems().catch(() => [])
  for (const l of listeners) l(items.length)
}

/** Subscribe to pending-count changes (for the UI badge). Returns an unsubscribe. */
export function subscribe(cb: Listener): () => void {
  listeners.add(cb)
  void allItems()
    .then((i) => cb(i.length))
    .catch(() => cb(0))
  return () => listeners.delete(cb)
}

export async function pendingCount(): Promise<number> {
  const items = await allItems().catch(() => [])
  return items.length
}

/** Queue a write for later delivery. Returns the item's idempotency key. */
export async function enqueue(
  method: string,
  path: string,
  body: unknown,
): Promise<string> {
  const item: OutboxItem = {
    id: uuid(),
    idempotencyKey: uuid(),
    method,
    path,
    body,
    createdAt: Date.now(),
    attempts: 0,
  }
  await put(STORE, item)
  await notify()
  // Opportunistic immediate flush if we're actually online.
  if (navigator.onLine) void flush()
  return item.idempotencyKey
}

function backoffMs(attempts: number): number {
  return Math.min(BASE_BACKOFF_MS * 2 ** attempts, MAX_BACKOFF_MS)
}

/** True for HTTP statuses where retrying is pointless (client errors). */
function isPermanent(status: number): boolean {
  return status >= 400 && status < 500 && status !== 408 && status !== 429
}

type Sender = (item: OutboxItem) => Promise<Response>

/**
 * Attempt to deliver every queued item in FIFO order.
 * `send` is injected so this module stays decoupled from the api client and is
 * unit-testable; `api.ts` passes a thin fetch wrapper that adds auth + the
 * `Idempotency-Key` header.
 */
export async function flush(send?: Sender): Promise<void> {
  if (flushing || !navigator.onLine) return
  flushing = true
  try {
    const doSend = send ?? defaultSender
    const items = await allItems()
    for (const item of items) {
      const due = item.createdAt + backoffMs(item.attempts)
      if (item.attempts > 0 && Date.now() < due) continue
      try {
        const res = await doSend(item)
        if (res.ok || res.status === 409 /* already applied (idempotent) */) {
          await remove(STORE, item.id)
        } else if (isPermanent(res.status)) {
          item.lastError = `HTTP ${res.status}`
          await put(DEAD, item)
          await remove(STORE, item.id)
        } else {
          item.attempts += 1
          item.lastError = `HTTP ${res.status}`
          if (item.attempts >= MAX_ATTEMPTS) {
            await put(DEAD, item)
            await remove(STORE, item.id)
          } else {
            await put(STORE, item)
          }
        }
      } catch (err) {
        item.attempts += 1
        item.lastError = err instanceof Error ? err.message : 'network'
        if (item.attempts >= MAX_ATTEMPTS) {
          await put(DEAD, item)
          await remove(STORE, item.id)
        } else {
          await put(STORE, item)
        }
        // Network still down — stop the pass, retry on next online event.
        break
      }
    }
  } finally {
    flushing = false
    await notify()
  }
}

const BASE = import.meta.env.VITE_API_URL ?? ''

const defaultSender: Sender = (item) => {
  let token: string | null = null
  try {
    token = localStorage.getItem('sv_token')
  } catch {
    token = null
  }
  return fetch(`${BASE}${item.path}`, {
    method: item.method,
    headers: {
      'Content-Type': 'application/json',
      'Idempotency-Key': item.idempotencyKey,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(item.body),
  })
}

/** Wire up automatic replay on reconnect. Call once at app start. */
export function initOutbox(): void {
  if (typeof window === 'undefined') return
  window.addEventListener('online', () => void flush())
  // Periodic safety-net sweep (covers flaky "online" events on mobile).
  window.setInterval(() => {
    if (navigator.onLine) void flush()
  }, 30_000)
  if (navigator.onLine) void flush()
}

/** List dead-lettered items for an admin review screen. */
export async function deadLetters(): Promise<OutboxItem[]> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const req = tx(db, DEAD, 'readonly').getAll()
    req.onsuccess = () => resolve(req.result as OutboxItem[])
    req.onerror = () => reject(req.error)
  })
}
