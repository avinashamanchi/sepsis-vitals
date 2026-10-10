/**
 * Router basename for the base path the build was made for.
 *
 * GitHub Pages builds use base "/sepsis-vitals/"; the Docker dashboard is
 * built with base "/". A hard-coded "/sepsis-vitals" basename made the
 * Docker dashboard render nothing at "/" (React Router renders null when the
 * URL does not start with the basename).
 */
export function routerBasename(baseUrl: string | undefined): string {
  const trimmed = (baseUrl ?? '/').replace(/\/+$/, '')
  return trimmed === '' ? '/' : trimmed
}
