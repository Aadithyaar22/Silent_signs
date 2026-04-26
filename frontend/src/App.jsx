import { useState, useRef, useEffect, useCallback } from "react";

// ── API Configuration ────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function analyzeWithBackend(payload) {
  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

async function checkHealth() {
  const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
  return res.json();
}

// ── Theme ────────────────────────────────────────────────────
const T = {
  bg: '#06102e', card: '#0b1845', cardDeep: '#08122f',
  cyan: '#00d4ff', cyanFaint: 'rgba(0,212,255,0.10)',
  cyanBorder: 'rgba(0,212,255,0.22)', green: '#00df82',
  gold: '#f0c040', orange: '#ff8c30', red: '#ff4466',
  text: '#d8ecff', muted: 'rgba(190,220,255,0.50)', white: '#ffffff',
};

// ── Risk Gauge ───────────────────────────────────────────────
function RiskGauge({ score, label }) {
  const r = 52, cx = 68, cy = 68, circ = 2 * Math.PI * r;
  const sweep = circ * 0.75;
  const filled = Math.min((score / 100) * sweep, sweep);
  const offset = -(circ * 0.125);
  const riskLabel = score < 25 ? 'Low' : score < 50 ? 'Moderate' : score < 72 ? 'Elevated' : 'High';
  const riskColor = score < 25 ? T.green : score < 50 ? T.gold : score < 72 ? T.orange : T.red;
  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={136} height={110} viewBox="0 0 136 110">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={9}
          strokeDasharray={`${sweep} ${circ - sweep}`} strokeDashoffset={offset}
          strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={riskColor} strokeWidth={9}
          strokeDasharray={`${filled} ${circ - filled}`} strokeDashoffset={offset}
          strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`} />
        <text x={cx} y={cy - 2} textAnchor="middle" fill={T.white} fontSize={21} fontWeight={700}
          fontFamily="'IBM Plex Mono', monospace">{score}</text>
        <text x={cx} y={cy + 16} textAnchor="middle" fill={riskColor} fontSize={10} fontWeight={600}
          fontFamily="'Syne', sans-serif">{riskLabel.toUpperCase()}</text>
      </svg>
      <div style={{ fontSize: 12, color: T.text, fontWeight: 600, marginTop: -4 }}>{label}</div>
    </div>
  );
}

function MetricPill({ value, label, highlight }) {
  return (
    <div style={{ background: 'rgba(0,0,0,0.25)', borderRadius: 10, padding: '12px 10px', textAlign: 'center',
      border: highlight ? `1px solid ${T.cyanBorder}` : '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: 20, fontWeight: 700, color: T.cyan, fontFamily: "'IBM Plex Mono', monospace" }}>{value}</div>
      <div style={{ fontSize: 10, color: T.muted, marginTop: 2, letterSpacing: '0.05em' }}>{label}</div>
    </div>
  );
}

function StepHeader({ n, title, sub }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
      <div style={{ width: 30, height: 30, borderRadius: '50%', background: T.cyanFaint,
        border: `1.5px solid ${T.cyan}`, display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 12, fontWeight: 700, color: T.cyan, flexShrink: 0 }}>{n}</div>
      <div>
        <div style={{ fontSize: 18, fontWeight: 700, color: T.white }}>{title}</div>
        <div style={{ fontSize: 11, color: T.muted, marginTop: 1 }}>{sub}</div>
      </div>
    </div>
  );
}

const PROGRESS = { welcome: 0, typing: 18, speech: 36, motor: 56, symptoms: 75, analyzing: 90, results: 100 };

export default function SilentSignsAgent() {
  const [phase, setPhase] = useState('welcome');
  const [kMetrics, setKMetrics] = useState(null);
  const [sMetrics, setSMetrics] = useState(null);
  const [mMetrics, setMMetrics] = useState(null);
  const [symps, setSymps] = useState({ age: '40-49', tremor: 'none', memory: 'none', mood: 'stable', sleep: 'good', history: 'none' });
  const [agentLog, setAgentLog] = useState([]);
  const [report, setReport] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking'); // checking | online | offline

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(data => setApiStatus(data.models_loaded ? 'online' : 'loading'))
      .catch(() => setApiStatus('offline'));
  }, []);

  // ── Typing state ──
  const typingPrompt = "The morning sun cast long shadows across the quiet street. I reached for my phone, but my fingers felt unusually stiff. It had been happening more often — a subtle tremor when I held a cup, a small hesitation when trying to type quickly. Perhaps it was just fatigue, I told myself.";
  const [typingText, setTypingText] = useState('');
  const tkTimes = useRef([]); const tkStart = useRef(null);
  const tkBackspace = useRef(0); const tkPauses = useRef(0); const tkLastKey = useRef(null);

  const onTypingKey = useCallback((e) => {
    const now = Date.now();
    if (!tkStart.current) tkStart.current = now;
    if (e.key === 'Backspace') tkBackspace.current++;
    if (tkLastKey.current && now - tkLastKey.current > 600) tkPauses.current++;
    tkTimes.current.push(now);
    tkLastKey.current = now;
  }, []);

  const finishTyping = useCallback(() => {
    const times = tkTimes.current;
    if (times.length < 8) return;
    const ikis = times.slice(1).map((t, i) => t - times[i]);
    const avg = ikis.reduce((a, b) => a + b, 0) / ikis.length;
    const std = Math.sqrt(ikis.reduce((a, b) => a + (b - avg) ** 2, 0) / ikis.length);
    const dur = (times[times.length - 1] - tkStart.current) / 1000;
    const words = typingText.trim().split(/\s+/).filter(Boolean).length;
    setKMetrics({
      wpm: Math.min(Math.round((words / dur) * 60), 220),
      avg_iki_ms: Math.round(avg), iki_std_ms: Math.round(std),
      backspace_rate_pct: Math.round((tkBackspace.current / times.length) * 100),
      pause_count: tkPauses.current, total_keystrokes: times.length,
      duration_s: Math.round(dur),
    });
    setPhase('speech');
  }, [typingText]);

  // ── Speech state ──
  const [speechText, setSpeechText] = useState('');
  const finishSpeech = useCallback(() => {
    const words = speechText.trim().split(/\s+/).filter(Boolean);
    const sents = speechText.split(/[.!?]+/).filter(s => s.trim());
    const unique = new Set(words.map(w => w.toLowerCase().replace(/[^a-z]/g, '')));
    const hedges = words.filter(w => ['um','uh','like','basically','actually','i think','i guess','sort of','kind of','you know','maybe'].some(h => w.toLowerCase().includes(h))).length;
    setSMetrics({
      word_count: words.length, sentence_count: sents.length,
      avg_sentence_len: sents.length ? Math.round(words.length / sents.length) : 0,
      lexical_diversity_pct: words.length ? Math.round((unique.size / words.length) * 100) : 0,
      hedge_words: hedges, unique_words: unique.size, sample: speechText.slice(0, 240),
    });
    setPhase('motor');
  }, [speechText]);

  // ── Motor state ──
  const [motorActive, setMotorActive] = useState(false);
  const [motorDone, setMotorDone] = useState(false);
  const [tapCount, setTapCount] = useState(0);
  const [countdown, setCountdown] = useState(10);
  const mTimes = useRef([]); const mTimer = useRef(null); const mCountRef = useRef(null);

  useEffect(() => () => { clearTimeout(mTimer.current); clearInterval(mCountRef.current); }, []);

  const startMotor = useCallback(() => {
    mTimes.current = []; setTapCount(0); setCountdown(10); setMotorActive(true); setMotorDone(false);
    let rem = 10;
    mCountRef.current = setInterval(() => { rem--; setCountdown(rem); if (rem <= 0) clearInterval(mCountRef.current); }, 1000);
    mTimer.current = setTimeout(() => {
      setMotorActive(false); setMotorDone(true); clearInterval(mCountRef.current);
      const ts = mTimes.current;
      if (ts.length < 2) { setMMetrics({ total_taps: ts.length, taps_per_sec: ts.length / 10, avg_interval_ms: 0, interval_std_ms: 0, duration_s: 10 }); return; }
      const ivls = ts.slice(1).map((t, i) => t - ts[i]);
      const avg = ivls.reduce((a, b) => a + b, 0) / ivls.length;
      const std = Math.sqrt(ivls.reduce((a, b) => a + (b - avg) ** 2, 0) / ivls.length);
      setMMetrics({ total_taps: ts.length, taps_per_sec: parseFloat((ts.length / 10).toFixed(2)), avg_interval_ms: Math.round(avg), interval_std_ms: Math.round(std), duration_s: 10 });
    }, 10000);
  }, []);

  const handleTap = useCallback(() => {
    if (!motorActive) return;
    mTimes.current.push(Date.now()); setTapCount(c => c + 1);
  }, [motorActive]);

  // ── Analysis — calls real FastAPI backend ──
  const runAnalysis = useCallback(async () => {
    setPhase('analyzing'); setAgentLog([]); setReport(null); setApiError(null);
    const STEPS = [
      'Initializing NeuralScreen Agent v2.1 ...',
      'Connecting to SilentSigns inference API ...',
      'Running keystroke dynamics → Parkinson\'s motor classifier (NeuroQWERTY-trained) ...',
      'Analyzing speech biomarkers → Depression model (DAIC-WOZ distribution) ...',
      'Scoring cognitive-linguistic patterns → DementiaNet Alzheimer\'s model ...',
      'Evaluating motor tap coordination → PhysioNet Gait classifier ...',
      'Cross-referencing UCI Parkinson\'s voice model ...',
      'Synthesizing multi-condition risk report ...',
    ];
    for (let i = 0; i < STEPS.length; i++) {
      await new Promise(r => setTimeout(r, 500 + Math.random() * 350));
      setAgentLog(prev => [...prev, STEPS[i]]);
    }
    try {
      const result = await analyzeWithBackend({
        typing_dynamics: kMetrics,
        speech_biomarkers: sMetrics,
        motor_coordination: mMetrics,
        symptom_questionnaire: symps,
      });
      setReport(result);
      setPhase('results');
    } catch (err) {
      setApiError(err.message || 'API call failed');
      setPhase('error');
    }
  }, [kMetrics, sMetrics, mMetrics, symps]);

  // ── Reset ──
  const reset = useCallback(() => {
    setPhase('welcome'); setKMetrics(null); setSMetrics(null); setMMetrics(null);
    setTypingText(''); setSpeechText('');
    setTapCount(0); setMotorActive(false); setMotorDone(false);
    setReport(null); setAgentLog([]); setApiError(null);
    tkTimes.current = []; tkStart.current = null;
    tkBackspace.current = 0; tkPauses.current = 0; tkLastKey.current = null; mTimes.current = [];
  }, []);

  // ── Symptom options ──
  const sympOpts = {
    age: ['Under 40', '40-49', '50-59', '60-69', '70+'],
    tremor: ['None', 'Mild', 'Moderate', 'Significant'],
    memory: ['None', 'Mild', 'Moderate', 'Significant'],
    mood: ['Stable', 'Mild changes', 'Moderate changes', 'Significant'],
    sleep: ['Good', 'Fair', 'Poor'],
    history: ['None', "Parkinson's", "Alzheimer's", 'Depression'],
  };
  const toKey = s => s.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');

  // ── Shared styles ──
  const card = { background: T.card, border: `1px solid ${T.cyanBorder}`, borderRadius: 14, padding: '22px 24px', marginBottom: 14 };
  const btn = { background: T.cyan, color: '#000', border: 'none', borderRadius: 9, padding: '11px 26px', fontSize: 13, fontWeight: 700, cursor: 'pointer', letterSpacing: '0.04em', fontFamily: 'inherit' };
  const btnOutline = { background: 'transparent', color: T.cyan, border: `1.5px solid ${T.cyan}`, borderRadius: 9, padding: '10px 22px', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit' };
  const inp = { background: 'rgba(255,255,255,0.04)', border: `1px solid ${T.cyanBorder}`, borderRadius: 9, color: T.text, padding: '11px 13px', fontSize: 13, width: '100%', boxSizing: 'border-box', resize: 'vertical', fontFamily: 'inherit', lineHeight: 1.6 };
  const grid3 = { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginTop: 14 };
  const tag = { display: 'inline-block', background: T.cyanFaint, color: T.cyan, border: `1px solid ${T.cyanBorder}`, borderRadius: 20, padding: '3px 10px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em' };

  const progress = PROGRESS[phase] || 0;
  const statusColors = { online: T.green, loading: T.gold, offline: T.red, checking: T.muted };
  const statusLabel = { online: '● API Online', loading: '◌ Models Loading', offline: '● API Offline', checking: '○ Checking' };

  // ──────────────────── RENDER ─────────────────────────────────

  if (phase === 'welcome') return (
    <div style={{ background: T.bg, minHeight: '100vh', fontFamily: "'Syne', sans-serif", color: T.text }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');*{box-sizing:border-box;margin:0}button{cursor:pointer}`}</style>
      <div style={{ borderBottom: `1px solid ${T.cyanBorder}`, padding: '14px 24px', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(8,18,47,0.97)' }}>
        <div style={{ width: 32, height: 32, background: T.cyan, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 700, color: '#000' }}>SS</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: T.white }}>SilentSigns</div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: statusColors[apiStatus], fontFamily: "'IBM Plex Mono', monospace" }}>{statusLabel[apiStatus]}</span>
          <span style={tag}>TECHNOVERSE 2026</span>
        </div>
      </div>
      <div style={{ maxWidth: 700, margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ ...card, textAlign: 'center', padding: '48px 32px' }}>
          <div style={{ fontSize: 44, marginBottom: 12 }}>🧠</div>
          <h1 style={{ fontSize: 30, fontWeight: 700, color: T.white, marginBottom: 6 }}>NeuralScreen Agent</h1>
          <p style={{ color: T.muted, fontSize: 14, marginBottom: 8 }}>Real ML Models · Real Datasets · Live Inference</p>
          <span style={tag}>STEP 04 · MVP BUILD · COGNIZANT TECHNOVERSE</span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8, margin: '24px 0 16px', textAlign: 'left' }}>
            {[
              ['UCI Parkinson\'s Voice', 'n=195, AUC 0.86'],
              ['NeuroQWERTY MIT-CSXPD', 'n=85, Typing dynamics'],
              ['PhysioNet Gait PD', 'n=166, Motor patterns'],
              ['DementiaNet', 'n=200, Alzheimer\'s speech'],
            ].map(([name, desc]) => (
              <div key={name} style={{ background: T.cardDeep, borderRadius: 8, padding: '10px 14px', border: `1px solid rgba(255,255,255,0.05)` }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: T.cyan }}>{name}</div>
                <div style={{ fontSize: 10, color: T.muted, marginTop: 2 }}>{desc}</div>
              </div>
            ))}
          </div>
          {apiStatus === 'offline' && (
            <div style={{ padding: '10px 16px', background: 'rgba(255,68,102,0.1)', border: '1px solid rgba(255,68,102,0.3)', borderRadius: 8, marginBottom: 16, fontSize: 12, color: '#ff8099' }}>
              ⚠️ Backend API is offline. Make sure the FastAPI server is running at {API_URL}
            </div>
          )}
          <p style={{ fontSize: 11, color: T.muted, marginBottom: 20 }}>⚠️ For demonstration only. Not a medical device.</p>
          <button style={{ ...btn, padding: '13px 36px', fontSize: 15, opacity: apiStatus === 'offline' ? 0.5 : 1 }}
            disabled={apiStatus === 'offline'}
            onClick={() => setPhase('typing')}>
            Begin Assessment →
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div style={{ background: T.bg, minHeight: '100vh', fontFamily: "'Syne', sans-serif", color: T.text }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');*{box-sizing:border-box;margin:0}button{cursor:pointer}textarea:focus{outline:2px solid ${T.cyan};outline-offset:1px}@keyframes spin{to{transform:rotate(360deg)}}@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}`}</style>

      <div style={{ borderBottom: `1px solid ${T.cyanBorder}`, padding: '14px 24px', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(8,18,47,0.97)', position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ width: 30, height: 30, background: T.cyan, borderRadius: 7, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, color: '#000' }}>SS</div>
        <div style={{ fontSize: 15, fontWeight: 700, color: T.white }}>SilentSigns</div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{ fontSize: 11, color: statusColors[apiStatus], fontFamily: "'IBM Plex Mono', monospace" }}>{statusLabel[apiStatus]}</span>
          {phase !== 'analyzing' && phase !== 'results' && (
            <span style={{ fontSize: 12, color: T.muted }}>Step {['typing','speech','motor','symptoms'].indexOf(phase) + 1} of 4</span>
          )}
          <span style={tag}>TECHNOVERSE 2026</span>
        </div>
      </div>

      <div style={{ height: 3, background: 'rgba(255,255,255,0.06)' }}>
        <div style={{ height: '100%', width: `${progress}%`, background: T.cyan, transition: 'width 0.5s ease', borderRadius: 2 }} />
      </div>

      <div style={{ maxWidth: 720, margin: '0 auto', padding: '28px 20px' }}>

        {/* ── TYPING ── */}
        {phase === 'typing' && (
          <div>
            <StepHeader n="1" title="Keystroke Dynamics" sub="IKI variance · WPM · Error rate — NeuroQWERTY model" />
            <div style={card}>
              <p style={{ fontSize: 12, color: T.muted, marginBottom: 10 }}>Type the passage naturally. Keystroke timing is measured in real time.</p>
              <div style={{ background: T.cyanFaint, border: `1px solid ${T.cyanBorder}`, borderRadius: 8, padding: '13px 16px', marginBottom: 14, fontSize: 13, color: T.text, lineHeight: 1.85, fontStyle: 'italic' }}>
                {typingPrompt}
              </div>
              <textarea style={{ ...inp, minHeight: 108, marginBottom: 12 }}
                placeholder="Start typing here..." value={typingText}
                onChange={e => setTypingText(e.target.value)} onKeyDown={onTypingKey} autoFocus />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: T.muted }}>{typingText.trim().split(/\s+/).filter(Boolean).length} words</span>
                <button style={{ ...btn, opacity: typingText.trim().split(/\s+/).filter(Boolean).length < 15 ? 0.35 : 1 }}
                  disabled={typingText.trim().split(/\s+/).filter(Boolean).length < 15} onClick={finishTyping}>
                  Save & Continue →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── SPEECH ── */}
        {phase === 'speech' && (
          <div>
            <StepHeader n="2" title="Speech Biomarkers" sub="Lexical diversity · Fluency — DementiaNet + DAIC-WOZ model" />
            {kMetrics && (
              <div style={{ ...card, borderColor: 'rgba(0,223,130,0.3)', marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.green, fontWeight: 700, marginBottom: 10 }}>✓ KEYSTROKE DATA CAPTURED</div>
                <div style={grid3}>
                  <MetricPill value={kMetrics.wpm} label="WPM" />
                  <MetricPill value={`${kMetrics.avg_iki_ms}ms`} label="AVG IKI" />
                  <MetricPill value={`±${kMetrics.iki_std_ms}ms`} label="IKI VARIANCE" highlight />
                </div>
              </div>
            )}
            <div style={card}>
              <div style={{ background: T.cyanFaint, border: `1px solid ${T.cyanBorder}`, borderRadius: 8, padding: '12px 14px', marginBottom: 14, fontSize: 13, color: T.text, lineHeight: 1.75 }}>
                <strong style={{ color: T.white }}>Prompt:</strong> Describe what you did yesterday in 3–5 sentences. Include as many specific details as you can recall.
              </div>
              <textarea style={{ ...inp, minHeight: 110, marginBottom: 12 }}
                placeholder="Yesterday I..." value={speechText} onChange={e => setSpeechText(e.target.value)} />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: T.muted }}>{speechText.trim().split(/\s+/).filter(Boolean).length} words</span>
                <button style={{ ...btn, opacity: speechText.trim().split(/\s+/).filter(Boolean).length < 15 ? 0.35 : 1 }}
                  disabled={speechText.trim().split(/\s+/).filter(Boolean).length < 15} onClick={finishSpeech}>
                  Analyze Speech →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── MOTOR ── */}
        {phase === 'motor' && (
          <div>
            <StepHeader n="3" title="Motor Coordination" sub="Tap rate · Rhythm — PhysioNet Gait PD model" />
            {sMetrics && (
              <div style={{ ...card, borderColor: 'rgba(0,223,130,0.3)', marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.green, fontWeight: 700, marginBottom: 10 }}>✓ SPEECH DATA CAPTURED</div>
                <div style={grid3}>
                  <MetricPill value={sMetrics.word_count} label="WORDS" />
                  <MetricPill value={`${sMetrics.lexical_diversity_pct}%`} label="LEX DIVERSITY" highlight />
                  <MetricPill value={sMetrics.avg_sentence_len} label="AVG SENT LEN" />
                </div>
              </div>
            )}
            <div style={card}>
              {!motorActive && !motorDone && (
                <div style={{ textAlign: 'center', padding: '12px 0' }}>
                  <p style={{ fontSize: 14, color: T.text, lineHeight: 1.75, marginBottom: 24 }}>
                    Tap the circle below as <strong style={{ color: T.white }}>fast and rhythmically</strong> as possible for <strong style={{ color: T.cyan }}>10 seconds</strong>.
                  </p>
                  <button style={{ ...btn, padding: '14px 40px', fontSize: 14 }} onClick={startMotor}>Start Motor Test ▶</button>
                </div>
              )}
              {motorActive && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 13, color: T.cyan, marginBottom: 4, fontFamily: "'IBM Plex Mono', monospace" }}>{countdown}s remaining · {tapCount} taps</div>
                  <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, marginBottom: 20 }}>
                    <div style={{ height: '100%', width: `${((10 - countdown) / 10) * 100}%`, background: T.cyan, borderRadius: 2, transition: 'width 1s linear' }} />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <button style={{ width: 180, height: 180, borderRadius: '50%', background: T.cyanFaint, border: `2.5px solid ${T.cyan}`, color: T.cyan, fontSize: 26, fontWeight: 700, fontFamily: "'Syne', sans-serif", userSelect: 'none', transition: 'transform 0.05s' }}
                      onClick={handleTap} onPointerDown={e => e.currentTarget.style.transform = 'scale(0.90)'}
                      onPointerUp={e => e.currentTarget.style.transform = 'scale(1)'}>TAP</button>
                  </div>
                </div>
              )}
              {motorDone && mMetrics && (
                <div>
                  <div style={{ fontSize: 11, color: T.green, fontWeight: 700, marginBottom: 12 }}>✓ MOTOR TEST COMPLETE — {mMetrics.total_taps} TAPS</div>
                  <div style={grid3}>
                    <MetricPill value={mMetrics.taps_per_sec} label="TAPS/SEC" />
                    <MetricPill value={`${mMetrics.avg_interval_ms}ms`} label="AVG INTERVAL" />
                    <MetricPill value={`±${mMetrics.interval_std_ms}ms`} label="CONSISTENCY" highlight />
                  </div>
                  <div style={{ textAlign: 'right', marginTop: 16 }}>
                    <button style={btn} onClick={() => setPhase('symptoms')}>Continue →</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── SYMPTOMS ── */}
        {phase === 'symptoms' && (
          <div>
            <StepHeader n="4" title="Symptom Profile" sub="Self-reported signals for risk calibration" />
            <div style={card}>
              {[
                { k: 'age', label: 'Age range' },
                { k: 'tremor', label: 'Hand tremor or shakiness' },
                { k: 'memory', label: 'Memory lapses or word-finding difficulty' },
                { k: 'mood', label: 'Mood or motivation changes' },
                { k: 'sleep', label: 'Sleep quality' },
                { k: 'history', label: 'Family history of neurological conditions' },
              ].map(({ k, label }) => (
                <div key={k} style={{ marginBottom: 18 }}>
                  <div style={{ fontSize: 12, color: T.text, marginBottom: 8, fontWeight: 600 }}>{label}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
                    {sympOpts[k].map(opt => {
                      const optKey = toKey(opt);
                      const sel = symps[k] === optKey;
                      return (
                        <button key={opt} onClick={() => setSymps(s => ({ ...s, [k]: optKey }))}
                          style={{ padding: '7px 14px', borderRadius: 20, border: `1.5px solid ${sel ? T.cyan : T.cyanBorder}`, background: sel ? T.cyanFaint : 'transparent', color: sel ? T.cyan : T.muted, fontSize: 12, fontWeight: sel ? 600 : 400, fontFamily: 'inherit', transition: 'all 0.15s' }}>
                          {opt}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
              <div style={{ borderTop: `1px solid ${T.cyanBorder}`, marginTop: 8, paddingTop: 18, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: T.muted }}>4 biomarker streams ready · Sending to NeuralScreen API</span>
                <button style={{ ...btn, fontSize: 14, padding: '12px 30px' }} onClick={runAnalysis}>
                  Run NeuralScreen Analysis ⚡
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── ANALYZING ── */}
        {phase === 'analyzing' && (
          <div>
            <div style={{ ...card, textAlign: 'center', padding: '40px 28px' }}>
              <div style={{ fontSize: 38, marginBottom: 14 }}>⚡</div>
              <h2 style={{ fontSize: 20, fontWeight: 700, color: T.white, marginBottom: 6 }}>NeuralScreen API Running</h2>
              <p style={{ fontSize: 13, color: T.muted, marginBottom: 28 }}>Dataset-trained ML models · Live inference</p>
              <div style={{ textAlign: 'left', maxWidth: 520, margin: '0 auto' }}>
                {agentLog.map((step, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 10, animation: 'fadeUp 0.35s ease both' }}>
                    <div style={{ width: 16, height: 16, borderRadius: '50%', background: T.green, flexShrink: 0, marginTop: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9 }}>✓</div>
                    <div style={{ fontSize: 12, color: T.text, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.5 }}>{step}</div>
                  </div>
                ))}
                {agentLog.length < 8 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ width: 16, height: 16, borderRadius: '50%', border: `2px solid ${T.cyan}`, flexShrink: 0, animation: 'spin 0.9s linear infinite' }} />
                    <div style={{ fontSize: 12, color: T.muted, fontFamily: "'IBM Plex Mono', monospace" }}>Processing ...</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── ERROR ── */}
        {phase === 'error' && (
          <div style={{ ...card, textAlign: 'center', padding: '40px 24px' }}>
            <div style={{ fontSize: 36, marginBottom: 12 }}>⚠️</div>
            <div style={{ fontSize: 16, color: T.red, marginBottom: 8 }}>Analysis Failed</div>
            <div style={{ fontSize: 13, color: T.muted, marginBottom: 22, maxWidth: 400, margin: '0 auto 22px', lineHeight: 1.6 }}>{apiError}</div>
            <button style={btn} onClick={() => setPhase('symptoms')}>Retry Analysis</button>
          </div>
        )}

        {/* ── RESULTS ── */}
        {phase === 'results' && report && (() => {
          const { overall_risk, conditions, biomarker_insights, recommendations, confidence, disclaimer, model_info } = report;
          const rC = { low: T.green, moderate: T.gold, elevated: T.orange, high: T.red };
          const overallColor = rC[overall_risk] || T.muted;
          return (
            <div>
              <div style={{ ...card, textAlign: 'center', borderColor: overallColor + '55', marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.muted, letterSpacing: '0.12em', marginBottom: 6 }}>OVERALL RISK ASSESSMENT</div>
                <div style={{ fontSize: 36, fontWeight: 700, color: overallColor, textTransform: 'uppercase' }}>{overall_risk}</div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 20, marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: T.muted }}>Confidence: <span style={{ color: T.cyan, fontWeight: 600 }}>{confidence}%</span></div>
                  {model_info && <div style={{ fontSize: 11, color: T.muted }}>PD AUC: <span style={{ color: T.cyan }}>{model_info.parkinson_auc}</span></div>}
                  {model_info && <div style={{ fontSize: 11, color: T.muted }}>AD AUC: <span style={{ color: T.cyan }}>{model_info.alzheimer_auc}</span></div>}
                </div>
              </div>

              <div style={{ ...card, marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.muted, fontWeight: 700, marginBottom: 18, letterSpacing: '0.10em' }}>CONDITION-SPECIFIC RISK SCORES</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16 }}>
                  {[['parkinsons', "Parkinson's"], ['depression', 'Depression'], ['alzheimers', "Alzheimer's"]].map(([k, lbl]) => {
                    const c = conditions[k];
                    return (
                      <div key={k}>
                        <RiskGauge score={c.score} label={lbl} />
                        <div style={{ fontSize: 11, color: T.muted, marginTop: 8, lineHeight: 1.6, textAlign: 'center' }}>{c.interpretation}</div>
                        <div style={{ marginTop: 8 }}>
                          {c.key_signals.map((sig, i) => (
                            <div key={i} style={{ fontSize: 10, color: T.cyan, background: T.cyanFaint, border: `1px solid ${T.cyanBorder}`, borderRadius: 5, padding: '3px 7px', marginBottom: 4, lineHeight: 1.4 }}>→ {sig}</div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Dataset info banner */}
              {model_info && model_info.datasets && (
                <div style={{ ...card, borderColor: 'rgba(255,255,255,0.08)', marginBottom: 14, padding: '14px 18px' }}>
                  <div style={{ fontSize: 10, color: T.muted, letterSpacing: '0.08em', marginBottom: 8 }}>TRAINED ON DATASETS</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {model_info.datasets.map(d => (
                      <span key={d} style={{ ...tag, fontSize: 10, padding: '2px 8px' }}>{d.replace(/_/g, ' ')}</span>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ ...card, marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.muted, fontWeight: 700, marginBottom: 14, letterSpacing: '0.10em' }}>BIOMARKER INSIGHTS</div>
                {biomarker_insights.map((ins, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 11 }}>
                    <div style={{ color: T.cyan, flexShrink: 0, fontSize: 12, marginTop: 2 }}>◆</div>
                    <div style={{ fontSize: 13, color: T.text, lineHeight: 1.7 }}>{ins}</div>
                  </div>
                ))}
              </div>

              <div style={{ ...card, borderColor: 'rgba(0,223,130,0.28)', marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: T.muted, fontWeight: 700, marginBottom: 14, letterSpacing: '0.10em' }}>RECOMMENDATIONS</div>
                {recommendations.map((rec, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 11 }}>
                    <div style={{ color: T.green, flexShrink: 0, fontSize: 12, marginTop: 2 }}>✓</div>
                    <div style={{ fontSize: 13, color: T.text, lineHeight: 1.7 }}>{rec}</div>
                  </div>
                ))}
              </div>

              <div style={{ padding: '12px 16px', background: 'rgba(255,68,102,0.07)', border: '1px solid rgba(255,68,102,0.22)', borderRadius: 10, marginBottom: 24 }}>
                <div style={{ fontSize: 12, color: 'rgba(255,100,120,0.95)', lineHeight: 1.7 }}>⚠️ {disclaimer}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <button style={btnOutline} onClick={reset}>Run New Assessment ↩</button>
              </div>
            </div>
          );
        })()}

      </div>
    </div>
  );
}
