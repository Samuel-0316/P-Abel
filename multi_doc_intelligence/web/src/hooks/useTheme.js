import { useState, useEffect, useCallback } from 'react';

/**
 * Dark/light theme state with localStorage persistence.
 * Applies `data-theme` attribute on <html>.
 */
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('p-abel-theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('p-abel-theme', theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggle };
}
