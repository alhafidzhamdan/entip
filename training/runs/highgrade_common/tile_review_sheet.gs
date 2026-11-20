/**
 * Google Apps Script that collects tile review ratings into a Google Sheet.
 *
 * The review page (rate_tiles.py build --endpoint <URL>) POSTs each rating here; every POST appends rows to the
 * sheet "ratings". A later rating of the same tile adds a new row, and the summary keeps the latest one.
 * GET with ?key=<READ_KEY> returns the whole sheet as CSV, for rate_tiles.py summary --sheet.
 *
 * Setup: see "Collecting ratings online" in README.md next to this file. Change READ_KEY before deploying, and
 * keep it out of the review page: anyone with the page can send ratings, but only the key reads them back.
 */
const READ_KEY = 'CHANGE-ME-to-a-long-random-string';
const SHEET = 'ratings';
const HEADER = ['received_at', 'rater', 'page_id', 'item', 'rating', 'note', 'seconds', 'rated_at', 'client'];
const RATINGS = ['mvp', 'necrosis', 'both', 'neither', 'artefact', 'other', 'unassessable', ''];

function sheet_() {
  const book = SpreadsheetApp.getActiveSpreadsheet();
  let sh = book.getSheetByName(SHEET);
  if (!sh) {
    sh = book.insertSheet(SHEET);
    sh.appendRow(HEADER);
    sh.setFrozenRows(1);
  }
  return sh;
}

// Text that starts like a formula is stored as text, so a typed note can never run in the sheet.
function cell_(v, max) {
  let s = v === undefined || v === null ? '' : String(v).slice(0, max || 2000);
  if (/^[=+\-@]/.test(s)) s = "'" + s;
  return s;
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const body = JSON.parse(e.postData.contents);
    const rows = (Array.isArray(body.rows) ? body.rows : [body]).slice(0, 1000);
    const now = new Date().toISOString();
    const out = [];
    rows.forEach(function (r) {
      if (!r || typeof r.item !== 'string' || RATINGS.indexOf(r.rating || '') < 0) return;
      out.push([now, cell_(r.rater, 20), cell_(r.page_id, 40), cell_(r.item, 20), cell_(r.rating, 20),
                cell_(r.note, 2000), cell_(r.seconds, 20), cell_(r.rated_at, 40), cell_(r.client, 40)]);
    });
    if (out.length) {
      const sh = sheet_();
      sh.getRange(sh.getLastRow() + 1, 1, out.length, HEADER.length).setNumberFormat('@').setValues(out);
    }
    return ContentService.createTextOutput(JSON.stringify({ok: true, saved: out.length}))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  if (!e.parameter.key || e.parameter.key !== READ_KEY || READ_KEY.indexOf('CHANGE-ME') === 0) {
    return ContentService.createTextOutput('forbidden');
  }
  const rows = sheet_().getDataRange().getDisplayValues();
  const csv = rows.map(function (r) {
    return r.map(function (c) {
      let s = String(c);
      if (s.charAt(0) === "'") s = s.slice(1);
      return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }).join(',');
  }).join('\n') + '\n';
  return ContentService.createTextOutput(csv).setMimeType(ContentService.MimeType.CSV);
}
