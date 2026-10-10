import { describe, expect, it } from 'vitest'

import { routerBasename } from '../lib/basePath'

describe('router basename (N39)', () => {
  it('follows the base the build was made for', () => {
    expect(routerBasename('/sepsis-vitals/')).toBe('/sepsis-vitals') // GitHub Pages
    expect(routerBasename('/')).toBe('/') // Docker dashboard: the app must render at "/"
    expect(routerBasename(undefined)).toBe('/')
  })
})
