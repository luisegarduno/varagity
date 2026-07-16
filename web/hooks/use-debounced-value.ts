"use client";

import { useEffect, useState } from "react";

/**
 * Debounce a fast-changing value (the ~80 ms anti-flash for streaming).
 *
 * The second sanctioned `useEffect`, and the only one that keeps a
 * dependency array. A debounce is a timer — an external system — whose
 * lifecycle is tied to the *value*, not to mount, so neither
 * {@link useMountEffect} nor a `key` remount can express it: remounting per
 * value would defeat the debounce it exists to provide. Encapsulating it in
 * a named primitive is the point of the rule; components consume this hook
 * instead of reaching for the timer themselves.
 */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  // eslint-disable-next-line no-restricted-syntax -- see the docstring
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
