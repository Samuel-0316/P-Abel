import { SunIcon, MoonIcon } from './Icons';

export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      aria-label="Toggle theme"
    >
      {theme === 'dark'
        ? <SunIcon size={15} />
        : <MoonIcon size={15} />
      }
    </button>
  );
}
