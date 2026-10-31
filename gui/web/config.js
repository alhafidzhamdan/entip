/* Where the GUI gets its data.
 *   mode "mock": synthetic patients and slides from ./mock, training runs in the browser.
 *   mode "http": central database (db/, `entip-db serve`) and training backend (training/, `entip-train serve`).
 * Add ?api=http to the address to try http mode without editing this file. */
window.ENTIP_CONFIG = {
  mode: /[?&]api=http\b/.test(location.search) ? "http" : "mock",
  base: "mock",
  dbBase: "http://127.0.0.1:8705",
  trainBase: "http://127.0.0.1:8100",
};
