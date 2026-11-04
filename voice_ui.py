"""The "Talk to us" tab: a small LiveKit web widget that connects the visitor's microphone to the voice agent."""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from config import Settings
from voice_token import make_token

LIVEKIT_CLIENT = "https://cdn.jsdelivr.net/npm/livekit-client@2.22.4/dist/livekit-client.umd.min.js"

WIDGET = """
<style>
  :root{--bg:#fff;--fg:#10212b;--muted:#4f6471;--line:#c9d6dc;--accent:#0b7a85;--accent-fg:#fff;--bad:#b3261e}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1a21;--fg:#e4eef2;--muted:#93aab5;--line:#25414d;--accent:#4cc3ce;--accent-fg:#06222a;--bad:#f2b8b5}}
  html,body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;font-size:15px}
  .box{display:flex;flex-direction:column;gap:12px;padding:4px 2px}
  button{font:inherit;font-weight:600;padding:12px 20px;border-radius:8px;border:0;background:var(--accent);color:var(--accent-fg);cursor:pointer;align-self:flex-start}
  button.end{background:transparent;color:var(--fg);border:1px solid var(--line)}
  button:disabled{opacity:.6;cursor:default}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  #status{color:var(--muted);min-height:1.4em}
  #status.bad{color:var(--bad)}
  #cap{display:flex;flex-direction:column;gap:6px;max-height:190px;overflow-y:auto;border-top:1px solid var(--line);padding-top:8px}
  #cap p{margin:0;line-height:1.4}
  #cap b{color:var(--muted);font-weight:600;margin-right:6px}
</style>
<div class="box">
  <button id="go">Start talking</button>
  <div id="status" role="status" aria-live="polite">Click the button, allow the microphone, then describe your problem.</div>
  <div id="cap" aria-label="Conversation transcript"></div>
</div>
<script src="__CLIENT__"></script>
<script>
const URL_ = __URL__, TOKEN = __TOKEN__;
const go = document.getElementById('go'), statusEl = document.getElementById('status'), cap = document.getElementById('cap');
let room = null, waitTimer = null;
const say = (t, bad) => { statusEl.textContent = t; statusEl.className = bad ? 'bad' : ''; };
function reset(msg, bad){ clearTimeout(waitTimer); room = null; go.textContent = 'Start talking'; go.className = ''; go.disabled = false; say(msg || 'Call ended.', bad); }
const lines = {};
// Speech-to-text sends many interim versions of the same sentence. They share a segment id, so each one replaces the text of one line.
function upsert(seg, who, text, final){
  if (!text) return;
  let l = lines[seg];
  if (l && l.final) return;
  if (!l) { const p = document.createElement('p'), b = document.createElement('b'), s = document.createElement('span'); b.textContent = who; p.appendChild(b); p.appendChild(s); cap.appendChild(p); l = lines[seg] = { s: s, final: false }; }
  l.s.textContent = text; l.final = !!final; cap.scrollTop = cap.scrollHeight;
}
go.onclick = async () => {
  if (room) { await room.disconnect(); return; }
  go.disabled = true; say('Connecting…');
  const { Room, RoomEvent } = LivekitClient;
  const r = new Room();
  r.on(RoomEvent.TrackSubscribed, (track) => { if (track.kind === 'audio') { const el = track.attach(); el.autoplay = true; document.body.appendChild(el); } });
  r.on(RoomEvent.ParticipantConnected, () => { clearTimeout(waitTimer); say('Connected. Tell us what went wrong.'); });
  r.on(RoomEvent.ActiveSpeakersChanged, (speakers) => { if (!room) return; const agent = speakers.some(s => s !== r.localParticipant); const me = speakers.some(s => s === r.localParticipant); if (agent) say('The assistant is speaking…'); else if (me) say('Listening…'); });
  r.on(RoomEvent.Disconnected, () => reset('Call ended. You can start another one.'));
  try {
    await r.connect(URL_, TOKEN);
    room = r;
    await r.startAudio();
    await r.localParticipant.setMicrophoneEnabled(true);
    go.textContent = 'End call'; go.className = 'end'; go.disabled = false;
    say('Waiting for the assistant…');
    if (r.registerTextStreamHandler) {
      r.registerTextStreamHandler('lk.transcription', async (reader, info) => {
        try {
          const attrs = (reader.info && reader.info.attributes) || {};
          const seg = attrs['lk.segment_id'] || (reader.info && reader.info.id) || String(Math.random());
          const final = attrs['lk.transcription_final'] === 'true';
          const who = info.identity === r.localParticipant.identity ? 'You' : 'Assistant';
          let text = '';
          for await (const chunk of reader) { text += chunk; upsert(seg, who, text, false); }
          upsert(seg, who, text, final);
        } catch (e) {}
      });
    }
    waitTimer = setTimeout(() => { if (room && r.remoteParticipants.size === 0) { say('Our voice assistant is not available right now. Please use the form instead.', true); } }, 25000);
  } catch (e) {
    reset(e && e.name === 'NotAllowedError' ? 'Microphone access was blocked. Allow it in your browser and try again.' : 'Could not start the call. Please try again or use the form.', true);
  }
};
</script>
"""


def widget_html(url: str, token: str) -> str:
    """json.dumps puts the values in as JavaScript strings; '<' is escaped so a value can never close the script tag."""
    def js(v: str) -> str:
        return json.dumps(v).replace("<", "\\u003c")
    return WIDGET.replace("__CLIENT__", LIVEKIT_CLIENT).replace("__URL__", js(url)).replace("__TOKEN__", js(token))


def render_talk(user, s: Settings) -> None:
    st.subheader("Talk to us", icon=":material/mic:")
    st.caption("Describe your problem out loud. Our voice assistant listens, raises a support request and tells you the answer and a reference number. "
               "Your voice is processed by LiveKit and our AI providers; only the text is saved.")
    if not s.livekit_enabled:
        st.info("Voice support is not set up yet. Please use the written form.", icon=":material/mic_off:")
        return
    try:
        token, _room = make_token(s, user.username, user.display_name)
    except RuntimeError:
        st.info("Voice support is not available right now. Please use the written form.", icon=":material/mic_off:")
        return
    components.html(widget_html(s.livekit_url, token), height=330)
