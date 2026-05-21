import { useState, useEffect, useRef } from 'react';

export function usePersistentState<T>(key: string, initialValue: T | (() => T)): [T, (val: T | ((prev: T) => T)) => void] {
  const [state, setState] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key);
      if (item !== null) {
        const parsed = JSON.parse(item);
        if (parsed !== null && parsed !== undefined) return parsed;
      }
    } catch (e) {
      console.error('Failed to read from localStorage:', e);
    }
    if (typeof initialValue === 'function') {
      return (initialValue as () => T)();
    }
    return initialValue;
  });

  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch (e) {
      console.error('Failed to write to localStorage:', e);
    }
  }, [key, state]);

  return [state, setState];
}
