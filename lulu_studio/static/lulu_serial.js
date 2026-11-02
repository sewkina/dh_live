// 噜噜 USB 连接（Chrome / Edge 的 Web Serial），协议见固件 kf_link.h：
// 电脑发 "@L <hex>\n"，噜噜回 "@R <hex>\n"，其余行是固件日志。
(function () {
  const CMD = { POSE: 0x01, TORQUE: 0x02, STREAM: 0x03, BUF_RESET: 0x04, BUF_APPEND: 0x05, BUF_INSTALL: 0x06,
    BUF_PLAY: 0x07, EXIT: 0x08, PING: 0x09, PLAY: 0x0a, LIST: 0x0b };
  const REP = { POSE: 0x81, ACK: 0x82, PONG: 0x89, LIST: 0x8b };
  const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
  const unhex = (s) => new Uint8Array((s.match(/../g) || []).map((h) => parseInt(h, 16)));
  const le16 = (v) => [v & 0xff, (v >> 8) & 0xff];
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  class LuluSerial {
    static supported() { return 'serial' in navigator; }

    constructor() {
      this.port = null; this.writer = null; this.version = null;
      this.waiters = []; this.onstatus = () => {}; this.keepalive = null;
    }

    get connected() { return !!this.port; }

    async connect() {
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      try { await port.setSignals({ dataTerminalReady: false, requestToSend: false }); } catch (e) { /* 有的驱动不支持 */ }
      this.port = port;
      this.writer = port.writable.getWriter();
      this._readLoop();
      this.version = await this.ping();
      if (!this.version) {
        await this.disconnect();
        throw new Error('噜噜没有回应。请确认插的是数据线、噜噜已开机，并且固件带关键帧播放器（lulu-kf）');
      }
      this.keepalive = setInterval(() => this._send([CMD.PING]).catch(() => {}), 5000);
      this.onstatus(true);
      return this.version;
    }

    async disconnect() {
      clearInterval(this.keepalive);
      const port = this.port;
      this.port = null;
      try { if (this.writer) { await this.writer.write(enc.encode('@L ' + hex([CMD.EXIT]) + '\n')); this.writer.releaseLock(); } } catch (e) {}
      try { if (this.reader) await this.reader.cancel(); } catch (e) {}
      try { if (port) await port.close(); } catch (e) {}
      this.writer = null; this.reader = null;
      this.onstatus(false);
    }

    async _readLoop() {
      let buf = '';
      while (this.port && this.port.readable) {
        this.reader = this.port.readable.getReader();
        try {
          for (;;) {
            const { value, done } = await this.reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let i;
            while ((i = buf.indexOf('\n')) >= 0) {
              const line = buf.slice(0, i).trim();
              buf = buf.slice(i + 1);
              if (line.startsWith('@R ')) this._onPayload(unhex(line.slice(3)));
            }
            if (buf.length > 4096) buf = buf.slice(-1024);
          }
        } catch (e) {
          break;
        } finally {
          try { this.reader.releaseLock(); } catch (e) {}
        }
      }
      if (this.port) { this.port = null; clearInterval(this.keepalive); this.onstatus(false); }
    }

    _onPayload(p) {
      if (!p.length) return;
      let msg = null;
      if (p[0] === REP.ACK && p.length >= 3) msg = { kind: 'ack', cmd: p[1], ok: !!p[2], text: dec.decode(p.slice(3)) };
      else if (p[0] === REP.PONG) msg = { kind: 'pong', text: dec.decode(p.slice(1)) };
      else if (p[0] === REP.LIST) msg = { kind: 'ack', cmd: CMD.LIST, ok: true, text: dec.decode(p.slice(1)) };
      if (!msg) return;
      this.waiters = this.waiters.filter((w) => !w(msg));
    }

    async _send(bytes) {
      if (!this.writer) throw new Error('还没连接噜噜');
      await this.writer.write(enc.encode('@L ' + hex(bytes) + '\n'));
    }

    _wait(match, timeout) {
      return new Promise((resolve) => {
        const t = setTimeout(() => { this.waiters = this.waiters.filter((w) => w !== fn); resolve(null); }, timeout);
        const fn = (m) => { if (!match(m)) return false; clearTimeout(t); resolve(m); return true; };
        this.waiters.push(fn);
      });
    }

    async _request(bytes, timeout = 2000) {
      const cmd = bytes[0];
      const p = this._wait((m) => m.kind === 'ack' && m.cmd === cmd, timeout);
      await this._send(bytes);
      const m = await p;
      if (!m) throw new Error('噜噜超时没有回应');
      if (!m.ok) throw new Error(m.text || '噜噜拒绝了这条指令');
      return m.text;
    }

    async ping(timeout = 2000) {
      const p = this._wait((m) => m.kind === 'pong', timeout);
      await this._send([CMD.PING]);
      const m = await p;
      return m ? m.text : null;
    }

    async _upload(action) {
      const data = enc.encode(JSON.stringify(action));
      if (data.length >= 4000) throw new Error('动作 JSON 超过 4000 字节，噜噜存不下');
      await this._request([CMD.BUF_RESET]);
      for (let k = 0; k < data.length; k += 160) {
        await this._request([CMD.BUF_APPEND, ...data.slice(k, k + 160)]);
      }
    }

    async install(action) { await this._upload(action); return this._request([CMD.BUF_INSTALL]); }

    async preview(action, intensity = 100, speed = 100) {
      await this._upload(action);
      return this._request([CMD.BUF_PLAY, ...le16(intensity), ...le16(speed)]);
    }

    async list() { return JSON.parse((await this._request([CMD.LIST])) || '[]'); }
  }

  window.LuluSerial = LuluSerial;
})();
