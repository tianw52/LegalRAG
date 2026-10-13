// localStorage wrappers that never throw (private mode / blocked storage).

export function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStored(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage unavailable — value lives for this session only */
  }
}
