/* Where the GUI gets its data.
 *   mode "mock": synthetic patients and slides from ./mock, training runs in the browser.
 *   mode "http": central database (P5) at dbBase and training backend (P2) at trainBase. */
window.ENTIP_CONFIG = {
  mode: "mock",
  base: "mock",
  dbBase: "/api/db",
  trainBase: "/api/train",
};
