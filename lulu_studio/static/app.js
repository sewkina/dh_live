(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------------------------------------------------------------- Telegram 小程序
  const TG = window.Telegram && window.Telegram.WebApp;
  const inTG = !!(TG && TG.initData);
  if (inTG) {
    TG.ready();
    TG.expand();
    document.documentElement.dataset.theme = TG.colorScheme === 'dark' ? 'dark' : 'light';
  }

  // ---------------------------------------------------------------- 小工具
  const store = {
    get(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 隐私模式 */ } },
  };
  let toastTimer;
  function toast(msg, err) {
    const t = $('#toast');
    t.textContent = msg;
    t.className = 'toast' + (err ? ' err' : '');
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, err ? 5000 : 2600);
  }
  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || ('出错了（' + r.status + '）'));
    return data;
  }
  const kb = (n) => (n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(0) + ' KB' : (n / 1048576).toFixed(1) + ' MB');
  const abs = (u) => new URL(u, location.href).href;

  function download(url, filename) {
    if (inTG) {
      if (TG.isVersionAtLeast && TG.isVersionAtLeast('8.0') && TG.downloadFile) {
        TG.downloadFile({ url: abs(url), file_name: filename });
      } else {
        TG.openLink(abs(url));
      }
      return;
    }
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ---------------------------------------------------------------- 动作预览（侧视图 + 俯视腰部）
  const U = 3000 / 270;
  const STAND = [-600, 600, -600, 600, 0];
  const toAngle = (i, o) => (i === 0 || i === 2 ? (o + 1000) / U : i === 1 || i === 3 ? (1000 - o) / U : o / U);

  function expandFrames(a) {
    const f = a.frames || [];
    const lp = a.loop;
    if (!lp || !(lp.times > 1)) return f.slice();
    const mid = f.slice(lp.from, lp.to + 1);
    let out = f.slice(0, lp.from);
    for (let k = 0; k < lp.times; k++) out = out.concat(mid);
    return out.concat(f.slice(lp.to + 1));
  }

  function makeTimeline(a) {
    const seq = expandFrames(a);
    const pts = [STAND].concat(seq.map((x) => x.p));
    const durs = seq.map((x) => x.d);
    const total = durs.reduce((s, d) => s + d, 0);
    const spline = a.interp === 'spline';
    return {
      total,
      at(t) {
        if (t >= total || !durs.length) return pts[pts.length - 1];
        let k = 0;
        while (t > durs[k]) { t -= durs[k]; k++; }
        const u = durs[k] ? t / durs[k] : 1;
        const p0 = pts[Math.max(0, k - 1)], p1 = pts[k], p2 = pts[k + 1], p3 = pts[Math.min(pts.length - 1, k + 2)];
        if (spline) {
          return p1.map((_, i) => 0.5 * ((2 * p1[i]) + (-p0[i] + p2[i]) * u + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * u * u + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * u * u * u));
        }
        const s = u * u * (3 - 2 * u);
        return p1.map((v, i) => v + (p2[i] - v) * s);
      },
    };
  }

  function dogSVG() {
    return `<svg viewBox="0 0 240 180" aria-hidden="true">
      <line x1="10" y1="152" x2="230" y2="152" stroke="currentColor" stroke-opacity=".15" stroke-width="2"/>
      <g class="far"><line class="l1" stroke="#c9b9a6" stroke-width="9" stroke-linecap="round"/><line class="l3" stroke="#c9b9a6" stroke-width="9" stroke-linecap="round"/></g>
      <g class="body">
        <rect x="58" y="66" width="70" height="38" rx="14" fill="#f2e6d6" stroke="#2b2f45" stroke-width="3"/>
        <g class="front"><rect x="118" y="62" width="64" height="42" rx="14" fill="#fff8ee" stroke="#2b2f45" stroke-width="3"/>
        <circle cx="178" cy="66" r="22" fill="#faf8f3" stroke="#2b2f45" stroke-width="4"/>
        <rect x="170" y="58" width="5" height="9" rx="2.5" fill="#263048"/><rect x="182" y="58" width="5" height="9" rx="2.5" fill="#263048"/></g>
      </g>
      <g class="near"><line class="l0" stroke="#2b2f45" stroke-width="9" stroke-linecap="round"/><line class="l2" stroke="#2b2f45" stroke-width="9" stroke-linecap="round"/></g>
      <g transform="translate(26 26)">
        <circle r="20" fill="currentColor" fill-opacity=".05"/>
        <rect x="-7" y="0" width="14" height="14" rx="4" fill="#f2e6d6" stroke="#2b2f45" stroke-width="2"/>
        <g class="yaw"><rect x="-7" y="-14" width="14" height="14" rx="4" fill="#fff8ee" stroke="#2b2f45" stroke-width="2"/><circle cx="0" cy="-15" r="3" fill="#e48b5a"/></g>
      </g>
    </svg>`;
  }

  function drawPose(svg, offsets) {
    const ang = offsets.map((o, i) => toAngle(i, o) * Math.PI / 180);
    const L = 58;
    const hips = [[160, 100], [156, 100], [84, 100], [80, 100]];
    const feet = hips.map(([x, y], i) => {
      const front = i < 2;
      return [x + (front ? 1 : -1) * Math.sin(ang[i]) * L, y + Math.cos(ang[i]) * L];
    });
    const lowest = Math.max(...feet.map((f) => f[1]));
    const dy = 150 - lowest;
    svg.querySelector('.body').setAttribute('transform', `translate(0 ${dy})`);
    for (let i = 0; i < 4; i++) {
      const l = svg.querySelector('.l' + i);
      l.setAttribute('x1', hips[i][0]); l.setAttribute('y1', hips[i][1] + dy);
      l.setAttribute('x2', feet[i][0]); l.setAttribute('y2', feet[i][1] + dy);
    }
    svg.querySelector('.yaw').setAttribute('transform', `rotate(${(-ang[4] * 180 / Math.PI).toFixed(1)})`);
  }

  const players = new Set();
  function attachPlayer(container, action, speed = 1) {
    container.innerHTML = dogSVG();
    const svg = container.querySelector('svg');
    const tl = makeTimeline(action);
    const p = { svg, tl, speed, t0: performance.now(), el: container };
    drawPose(svg, STAND);
    players.add(p);
    return p;
  }
  function tick(now) {
    for (const p of players) {
      if (!p.el.isConnected) { players.delete(p); continue; }
      const cycle = p.tl.total + 900; // 播完停一下再重播
      const t = ((now - p.t0) * p.speed) % cycle;
      drawPose(p.svg, t <= p.tl.total ? p.tl.at(t) : p.tl.at(p.tl.total));
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // ---------------------------------------------------------------- 声音
  let audio = null;
  function playSound(url, btn) {
    if (audio && !audio.paused && audio.dataset.src === url) { audio.pause(); return; }
    if (audio) audio.pause();
    $$('.play').forEach((b) => { b.textContent = '▶'; });
    audio = new Audio(url);
    audio.dataset.src = url;
    if (btn) { btn.textContent = '❚❚'; audio.onended = audio.onpause = () => { btn.textContent = '▶'; }; }
    audio.play().catch(() => toast('这个浏览器放不了这种音频格式', true));
  }

  // ---------------------------------------------------------------- USB 连接噜噜
  const serial = window.LuluSerial && LuluSerial.supported() && !inTG ? new LuluSerial() : null;
  const usbBtn = $('#usbBtn');
  if (serial) {
    usbBtn.hidden = false;
    serial.onstatus = (on) => {
      usbBtn.textContent = on ? '已连接噜噜' : '连接噜噜';
      usbBtn.classList.toggle('on', on);
      if (!on) toast('和噜噜的 USB 连接断开了');
    };
    usbBtn.onclick = async () => {
      if (serial.connected) { await serial.disconnect(); return; }
      try {
        const v = await serial.connect();
        toast('已连接噜噜（' + v + '）');
      } catch (e) {
        if (e.name !== 'NotFoundError') toast(e.message || String(e), true);
      }
    };
  }

  // ---------------------------------------------------------------- 列表
  const TAB_HINT = {
    actions: '点开可以看动画预览、下载 JSON。电脑上用 Chrome 插 USB 连上噜噜后，可以一键装进噜噜，装好后大模型就能按动作名叫它做。',
    emotes: '噜噜屏幕上的表情（240×240 圆屏）。点开可以听配套音效、下载 .eaf。换进噜噜需要重新烧录资源分区。',
    voices: '噜噜说话的声音。点 ▶ 试听，下载里有试听音频和后端配置片段。',
  };
  let tab = 'actions';
  let items = [];
  let cfg = { upload_needs_token: false };

  async function load() {
    try {
      items = await api('/api/items?type=' + tab);
    } catch (e) {
      items = [];
      toast(e.message, true);
    }
    render();
  }

  function filtered() {
    const q = $('#search').value.trim().toLowerCase();
    const src = $('#sourceFilter').value;
    return items.filter((i) => (!src || i.source === src) &&
      (!q || (i.name + i.desc + i.id + i.author + i.tags.join(' ')).toLowerCase().includes(q)));
  }

  function stageHTML(it) {
    if (it.type === 'emotes') {
      return `<div class="stage"><div class="screen">${it.preview ? `<img loading="lazy" src="${esc(it.preview)}" alt="">` : ''}</div></div>`;
    }
    if (it.type === 'voices') {
      return `<div class="stage voice-stage"><button class="play" data-sound="${esc(it.sound || '')}" aria-label="试听">▶</button>
        <div class="sample">${esc(it.sample_text || '')}</div></div>`;
    }
    return '<div class="stage"></div>';
  }

  function render() {
    const grid = $('#grid');
    const list = filtered();
    grid.innerHTML = '';
    $('#empty').hidden = list.length > 0;
    $('#tabHint').textContent = TAB_HINT[tab] || '';
    const pack = $('#packBtn');
    pack.onclick = (e) => { e.preventDefault(); download('/api/pack/' + tab + '.zip', 'lulu_' + tab + '_pack.zip'); };
    for (const it of list) {
      const card = document.createElement('div');
      card.className = 'card item';
      card.tabIndex = 0;
      card.innerHTML = `${stageHTML(it)}
        <div class="name">${esc(it.name)}</div>
        <div class="meta">${it.source === 'official' ? '<span class="badge">官方</span>' : '<span class="badge up">上传</span>'}
          ${it.tags.map((t) => `<span>#${esc(t)}</span>`).join('')}<span>↓${it.downloads}</span></div>`;
      if (it.type === 'actions' && it.action) attachPlayer(card.querySelector('.stage'), it.action);
      card.addEventListener('click', (e) => {
        const b = e.target.closest('.play');
        if (b) { e.stopPropagation(); playSound(b.dataset.sound, b); return; }
        openDetail(it);
      });
      card.addEventListener('keydown', (e) => { if (e.key === 'Enter') openDetail(it); });
      grid.appendChild(card);
    }
  }

  // ---------------------------------------------------------------- 详情
  function openSheet(id) { $(id).hidden = false; if (inTG && TG.BackButton) TG.BackButton.show(); }
  function closeSheets() {
    $$('.sheet').forEach((s) => { s.hidden = true; });
    if (audio) audio.pause();
    if (inTG && TG.BackButton) TG.BackButton.hide();
  }
  $$('[data-close]').forEach((el) => el.addEventListener('click', closeSheets));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheets(); });
  if (inTG && TG.BackButton) TG.BackButton.onClick(closeSheets);

  function fileName(url) { return url ? url.split('/').pop() : ''; }

  function openDetail(it) {
    const c = $('#sheetContent');
    const extra = [];
    let buttons = '';
    if (it.type === 'actions') {
      const a = it.action || {};
      const secs = (makeTimeline(a).total / 1000).toFixed(1);
      extra.push(`<p class="kv">动作名 <code>${esc(a.name)}</code> · ${a.frames ? a.frames.length : 0} 帧 · 约 ${secs} 秒 · ${a.interp === 'spline' ? '连续' : '每帧停顿'}${a.emotion ? ' · 配表情 ' + esc(a.emotion) : ''}</p>`);
      buttons += `<a class="btn primary" data-dl="${esc(it.main)}?dl=1" data-fn="${esc(fileName(it.main))}">下载动作 JSON</a>`;
      if (serial) {
        buttons += '<button class="btn" data-act="install">装进噜噜（USB）</button>';
        buttons += '<button class="btn ghost" data-act="try">在噜噜上试播</button>';
      }
      extra.push(serial
        ? '<p class="hint">“装进噜噜”只保存不动；装好后后端大模型可以用 self.dog.play 按动作名调用。“试播”会让噜噜马上动起来，请先把它放在平地上。</p>'
        : `<p class="hint">${inTG ? '在 Telegram 里不能连 USB。' : '这个浏览器不支持 USB 连接。'}要装进噜噜，请在电脑上用 Chrome 或 Edge 打开这个网站，插上数据线。</p>`);
      extra.push(`<details><summary class="kv">看 JSON</summary><pre>${esc(JSON.stringify(a, null, 1))}</pre></details>`);
    } else if (it.type === 'emotes') {
      extra.push(`<p class="kv">表情名 <code>${esc(it.emote || '')}</code> · ${it.fps || 10} 帧/秒 · ${kb(it.size)}</p>`);
      if (it.sound) buttons += `<button class="btn" data-sound="${esc(it.sound)}">▶ 听音效</button>`;
      buttons += `<a class="btn primary" data-dl="${esc(it.main)}?dl=1" data-fn="${esc(fileName(it.main))}">下载 .eaf</a>`;
      extra.push('<div class="warn">表情存在噜噜的资源分区里，换进去要在电脑上重新编译并烧录资源，暂时不能在网页上一键换。下载后交给刷机流程就行。</div>');
    } else {
      buttons += `<button class="btn" data-sound="${esc(it.sound)}">▶ 试听</button>`;
      buttons += `<a class="btn primary" data-dl="${esc(it.sound)}?dl=1" data-fn="${esc(fileName(it.sound))}">下载试听音频</a>`;
      if (it.main) buttons += `<a class="btn" data-dl="${esc(it.main)}?dl=1" data-fn="${esc(it.id)}_preset.json">下载配置</a>`;
      if (it.sample_text) extra.push(`<p class="kv">试听这句：${esc(it.sample_text)}</p>`);
      extra.push('<p class="hint">配置是后端 config 里 TTS 那一段要改的值。换音色要改后端配置并重启后端。</p>');
      extra.push('<details><summary class="kv">看配置</summary><pre class="preset">读取中…</pre></details>');
    }
    buttons += `<a class="btn ghost" data-dl="${esc(it.zip)}" data-fn="lulu_${esc(it.type)}_${esc(it.id)}.zip">打包下载（${kb(it.size)}）</a>`;
    if (it.source === 'upload') buttons += '<button class="btn ghost" data-act="delete">删除</button>';

    c.innerHTML = `<div class="detail">
      <h2>${esc(it.name)}</h2>
      <div class="meta kv">${it.source === 'official' ? '<span class="badge">官方</span>' : '<span class="badge up">上传</span>'}
        ${it.author ? '作者 ' + esc(it.author) + ' · ' : ''}下载 ${it.downloads} 次</div>
      ${stageHTML(it)}
      ${it.type === 'actions' ? '<div class="speed">预览速度 <button class="chip" data-speed="0.5">慢</button><button class="chip on" data-speed="1">正常</button><button class="chip" data-speed="2">快</button></div>' : ''}
      ${it.desc ? `<p>${esc(it.desc)}</p>` : ''}
      ${extra.join('')}
      <div class="actions-row">${buttons}</div>
      <p class="hint confirm" hidden></p>
    </div>`;

    let player = null;
    if (it.type === 'actions' && it.action) player = attachPlayer($('.stage', c), it.action);
    $$('[data-speed]', c).forEach((b) => b.addEventListener('click', () => {
      $$('[data-speed]', c).forEach((x) => x.classList.toggle('on', x === b));
      if (player) { player.speed = parseFloat(b.dataset.speed); player.t0 = performance.now(); }
    }));
    $$('[data-sound]', c).forEach((b) => b.addEventListener('click', () => playSound(b.dataset.sound, b.classList.contains('play') ? b : null)));
    $$('[data-dl]', c).forEach((b) => b.addEventListener('click', (e) => { e.preventDefault(); download(b.dataset.dl, b.dataset.fn); }));
    if (it.type === 'voices' && it.main) {
      fetch(it.main).then((r) => r.text()).then((t) => { $('.preset', c).textContent = t; }).catch(() => {});
    }
    const confirmP = $('.confirm', c);
    $$('[data-act]', c).forEach((b) => b.addEventListener('click', async () => {
      const act = b.dataset.act;
      if (act === 'delete') return confirmThen(b, confirmP, '再点一次“确认删除”，会移到电脑上的回收站文件夹。', '确认删除', () => del(it));
      if (!serial.connected) {
        try { await serial.connect(); } catch (e) { if (e.name !== 'NotFoundError') toast(e.message || String(e), true); return; }
      }
      if (act === 'try') {
        return confirmThen(b, confirmP, '噜噜会马上动起来，请先放在平地上，再点一次“确认试播”。', '确认试播', async () => {
          try { await serial.preview(it.action); toast('噜噜开始做“' + it.name + '”'); } catch (e) { toast(e.message, true); }
        });
      }
      b.disabled = true;
      try {
        await serial.install(it.action);
        toast('已装进噜噜：' + it.action.name);
      } catch (e) {
        toast('没装上：' + e.message, true);
      } finally { b.disabled = false; }
    }));
    openSheet('#sheet');
  }

  function confirmThen(btn, p, text, label, fn) {
    if (btn.dataset.armed) {
      delete btn.dataset.armed;
      btn.textContent = btn.dataset.label;
      p.hidden = true;
      return fn();
    }
    btn.dataset.armed = '1';
    btn.dataset.label = btn.textContent;
    btn.textContent = label;
    p.textContent = text;
    p.hidden = false;
    setTimeout(() => { if (btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = btn.dataset.label; p.hidden = true; } }, 8000);
  }

  async function del(it) {
    try {
      await api(`/api/items/${it.type}/${it.id}`, { method: 'DELETE', headers: { 'X-Lulu-Token': store.get('lulu_token') } });
      toast('已删除');
      closeSheets();
      load();
    } catch (e) { toast(e.message, true); }
  }

  // ---------------------------------------------------------------- 上传
  const upType = $('#upType');
  function syncUploadFields() {
    $$('[data-for]').forEach((el) => {
      const on = el.dataset.for === upType.value;
      el.hidden = !on;
      $$('input, select, textarea', el).forEach((i) => { i.disabled = !on; });
    });
  }
  upType.addEventListener('change', syncUploadFields);
  syncUploadFields();

  $('#uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const msg = $('#upMsg');
    const fd = new FormData(form);
    for (const [k, v] of Array.from(fd.entries())) {
      if (v instanceof File && !v.name) fd.delete(k);
    }
    const btn = form.querySelector('[type=submit]');
    btn.disabled = true;
    msg.textContent = '上传中…';
    try {
      const it = await api('/api/upload', { method: 'POST', body: fd, headers: { 'X-Lulu-Token': store.get('lulu_token') } });
      msg.textContent = '';
      form.reset();
      syncUploadFields();
      toast('上传成功：' + it.name);
      switchTab(it.type);
      setTimeout(() => openDetail(it), 300);
    } catch (err) {
      msg.textContent = err.message;
      if (/口令/.test(err.message)) openSettings();
    } finally { btn.disabled = false; }
  });

  // ---------------------------------------------------------------- 设置
  function openSettings() {
    $('#tokenInput').value = store.get('lulu_token');
    $('#tokenHint').textContent = cfg.upload_needs_token
      ? '这台电脑设了上传口令，上传和删除前要填。'
      : '这台电脑没设口令，谁能打开网站谁就能上传。开公网链接（给 Telegram 用）前，请在电脑上设 LULU_STUDIO_TOKEN。';
    openSheet('#settings');
  }
  $('#settingsBtn').addEventListener('click', openSettings);
  $('#saveToken').addEventListener('click', () => { store.set('lulu_token', $('#tokenInput').value.trim()); closeSheets(); toast('已保存'); });

  // ---------------------------------------------------------------- 切换分类
  function switchTab(t) {
    tab = t;
    $$('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
    $('#listView').hidden = t === 'upload';
    $('#uploadView').hidden = t !== 'upload';
    if (t !== 'upload') load();
  }
  $$('.tabs button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
  $('#search').addEventListener('input', render);
  $('#sourceFilter').addEventListener('change', render);

  api('/api/config').then((c) => { cfg = c; }).catch(() => {});
  switchTab('actions');
})();
