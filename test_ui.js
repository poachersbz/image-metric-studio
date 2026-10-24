// Run with: node test_ui.js. Executes the actual page handlers with small UI stubs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'static/index.html'), 'utf8');

function between(start, end) {
  const from = html.indexOf(start);
  const to = html.indexOf(end, from + start.length);
  assert(from >= 0 && to > from, `Missing source markers: ${start}, ${end}`);
  return html.slice(from, to);
}

function element() {
  return { style: {}, classList: { remove() {}, add() {} }, addEventListener() {} };
}

function testLastPairRemoval() {
  const ctx = {
    jobs: [], activeJobId: null, jobCounter: 0, currentSessionId: 'old-session',
    activeRequestController: null, volumeRequestController: null, volumeChartInstance: null,
    renderQueue() {}, updateFileLabel() {}, checkInputs() {}, setStatus() {}, setExportHint() {},
  };
  for (const name of `successBanner warningBanner errorBanner metaContainer sliderCard
    sliceRuler imgPreviewA placeholderA imgPreviewB placeholderB imgDiffMap placeholderDiff
    dlA dlB dlDiff sliceMetricsBody volumeMetricsBody chartHint btnRunVolume loadingOverlay
    btnCompareLabel sessionChip btnSaveCombined btnSaveAllSlices roiRectA roiRectB roiRectDiff
    roiBadgeA fileName1 iconA fileName2 iconB`.split(/\s+/)) ctx[name] = element();
  vm.createContext(ctx);
  vm.runInContext(between('  function createJob()', '  function renderQueue()'), ctx);
  vm.runInContext(between('  function clearWorkspace()', "  btnAddPair.addEventListener"), ctx);
  vm.runInContext(`const job = createJob(); activeJobId = job.id;
    job.sessionId = currentSessionId; removeJob(job.id);`, ctx);
  assert.equal(ctx.currentSessionId, null);
  assert.equal(ctx.jobs.length, 1);
  assert.equal(ctx.jobs[0].sessionId, null);
  assert.equal(ctx.activeJobId, ctx.jobs[0].id);
}

async function testExportSnapshot() {
  const requests = [];
  const ctx = {
    FormData, sliceSlider: { max: '2' }, currentSessionId: 'pair-A',
    getActiveJob: () => ({}), getNormalize: () => false, getResample: () => false,
    getBrain: () => false, getRegister: () => false, selectRegisterType: { value: 'rigid' },
    selectDiffMode: { value: 'signed' }, selectDiffCmap: { value: 'gray' },
    appendWindowParams(fd) { fd.append('window_center', '10'); }, appendRoiParams() {},
    btnSaveAllSlices: { addEventListener(event, fn) { ctx.run = fn; } },
    btnSaveCombined: {}, setExportHint() {},
    window: { async showDirectoryPicker() {
      // Switching during the folder picker must not change the export either.
      ctx.currentSessionId = 'pair-B';
      ctx.selectDiffMode.value = 'absolute';
      return { async getFileHandle() { return { async createWritable() {
        return { async write() {}, async close() {} };
      } }; } };
    } },
    async fetch(url, { body }) {
      requests.push(Object.fromEntries(body));
      ctx.currentSessionId = 'pair-C';
      return { ok: true, async blob() { return new Blob(); } };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(between('  function buildPlotFormData(', '  function setExportHint('), ctx);
  vm.runInContext(between("  btnSaveAllSlices.addEventListener", '  // ---------- Chart metric toggles'), ctx);
  await ctx.run();
  assert.equal(requests.length, 3);
  requests.forEach((request, i) => {
    assert.equal(request.session_id, 'pair-A');
    assert.equal(request.slice_idx, String(i));
    assert.equal(request.diff_mode, 'signed');
    assert.equal(request.window_center, '10');
  });
  assert.equal(ctx.btnSaveAllSlices.disabled, false);
}

async function testMprControls() {
  const requests = [];
  const handlers = {};
  const input = name => ({ value: '10', addEventListener(event, fn) { handlers[name] = fn; } });
  const ctx = {
    FormData, viewSessionId: 'volume-A', viewMprActive: true, viewCurrentSlice: 2,
    viewMaxSlices: 10, viewSliceInFlight: false, viewSlicePending: null, mprReqSeq: 0,
    mprNormX: 0.5, mprNormY: 0.5,
    viewSliceSlider: input('slice'), viewSliceCounter: {},
    viewWlSlider: input('wl'), viewWwSlider: input('ww'),
    viewWlValue: {}, viewWwValue: {}, viewAutoWl: input('auto'),
    btnViewResetWindow: input('reset'),
    setTimeout(fn) { fn(); }, clearTimeout() {},
    showViewSlice() {}, setViewWindowSliders(win) { ctx.lastWindow = win; },
    positionMprCrosshairs() {}, console,
    imgMprAxial: element(), imgMprCoronal: element(), imgMprSagittal: element(),
    mprCoordZ: {}, mprCoordY: {}, mprCoordX: {}, statIntensity: {},
    async fetch(url, { body }) {
      requests.push({ url, params: Object.fromEntries(body) });
      return { ok: true, async json() { return {
        axial: 'axial', coronal: 'coronal', sagittal: 'sagittal', value: 1,
        coords: { z: Number(body.get('z') ?? body.get('slice_idx')), y: 4, x: 4 },
        dims: { z: 10, y: 8, x: 8 },
        window: { center: 20, width: 40 },
      }; } };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(between('  async function fetchViewSlice(', "  btnViewPlay.addEventListener"), ctx);
  vm.runInContext(between('  async function fetchMprViews(', '  function positionMprCrosshairs('), ctx);
  handlers.wl(); handlers.ww(); handlers.reset(); handlers.auto();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 4);
  assert(requests.every(request => request.url === '/api/view_mpr'));
  assert.equal(requests[0].params.window_center, '10');
  assert.equal(requests[2].params.window_center, undefined);
  assert.equal(ctx.lastWindow.center, 20);
  assert.equal(ctx.imgMprAxial.src, 'axial');
  await ctx.fetchViewSlice(5);
  assert.equal(requests.at(-1).params.z, '5');
  assert.equal(ctx.viewSliceSlider.value, 5);
  assert.equal(ctx.viewSliceCounter.textContent, 'SLICE 06 / 10');
  ctx.viewMprActive = false;
  await ctx.fetchViewSlice(6);
  assert.equal(requests.at(-1).url, '/api/view_slice');
}

(async () => {
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1]);
  }
  testLastPairRemoval();
  await testExportSnapshot();
  await testMprControls();
  console.log('OK: page syntax, pair removal, export snapshot, MPR controls');
})().catch(error => { console.error(error); process.exitCode = 1; });
