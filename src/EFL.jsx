import React, { useEffect, useRef, useState } from "react";

/**
 * Elemental Field Lab — Fluids/Smoke/Wind/Plasma (v3.0.0 — Phase 0–2)
 *
 * PHASE 0: Dev HUD & Guardrails
 *  • Per-frame stats (max|v|, avg|div| pre/post, substeps, NaN count, FPS)
 *  • Shader/FBO validation overlay; null-guarded programs; safe teardown
 *
 * PHASE 1: Core Upgrades
 *  • MacCormack advection (vel/dye/temp) to reduce numerical smearing
 *  • CFL-based substepping for stability at high velocities
 *
 * PHASE 2: Element Plugins (distinct physics)
 *  • Fluid: light viscosity blur + curvature (surface-tension proxy) from dye
 *  • Smoke: ambient temperature gradient & top evaporation (stratification)
 *  • Plasma: vector potential A, pseudo-MHD forces via J×B
 *
 * This file is PART 1 of 2 (Canvas line-limit split).
 * PART 1 ends right before the JSX return(). PART 2 will start at the return(
 * and include the draw/step implementations and helpers.
 */

// ---------- Tiny helpers ----------
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const onceWarn = ((set)=>{ const seen=new Set(); return (k,msg)=>{ if(seen.has(k)) return; seen.add(k); console.warn(msg); }; })();

const vertSrc = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main(){
  vUv = aPos*0.5+0.5;
  gl_Position = vec4(aPos,0.0,1.0);
}`;

function makeGL(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
  if (!gl) throw new Error("WebGL2 not supported");
  const ext = gl.getExtension('EXT_color_buffer_float');
  if (!ext) throw new Error('EXT_color_buffer_float not supported');
  return gl;
}

function compile(gl, type, src, label) {
  if (!src || typeof src !== 'string') throw new Error(`Shader source missing for ${label||'unnamed'}`);
  const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) || 'shader compile error';
    const msg = `[GLSL COMPILE ERROR] ${label||''}\n${log}`;
    console.error(msg); gl.deleteShader(sh); throw new Error(msg);
  }
  return sh;
}

function makeProgram(gl, fsSrc, defines="", label="fs") {
  const vs = compile(gl, gl.VERTEX_SHADER, vertSrc, 'vertex');
  const fs = compile(gl, gl.FRAGMENT_SHADER, `#version 300 es\nprecision highp float;\n${defines}\n${fsSrc}`, label);
  const pr = gl.createProgram(); gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.linkProgram(pr);
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(pr) || 'program link error';
    console.error(`[GLSL LINK ERROR] ${label}\n${log}`);
    gl.deleteShader(vs); gl.deleteShader(fs); gl.deleteProgram(pr);
    throw new Error(log);
  }
  gl.validateProgram(pr);
  if (!gl.getProgramParameter(pr, gl.VALIDATE_STATUS)) {
    const vlog = gl.getProgramInfoLog(pr) || 'program validate error';
    console.error(`[GLSL VALIDATE ERROR] ${label}\n${vlog}`);
    gl.deleteShader(vs); gl.deleteShader(fs); gl.deleteProgram(pr);
    throw new Error(vlog);
  }
  gl.deleteShader(vs); gl.deleteShader(fs);
  return pr;
}

function makeQuad(gl){
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  const arr = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null); gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return vao;
}

function tex(gl, w, h, internalFormat, format, type, filter) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return t;
}

function fbo(gl, colorTex) {
  const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (!ok) throw new Error('FBO incomplete');
  return fb;
}

function doubleFBO(gl, w, h, internalFormat, format, type, filter){
  const a = tex(gl,w,h,internalFormat,format,type,filter); const fa = fbo(gl,a);
  const b = tex(gl,w,h,internalFormat,format,type,filter); const fb_ = fbo(gl,b);
  return { w,h, read:a, write:b, fboR:fa, fboW:fb_, swap(){ const t=this.read; this.read=this.write; this.write=t; const f=this.fboR; this.fboR=this.fboW; this.fboW=f; } };
}

function singleFBO(gl, w, h, internalFormat, format, type, filter){
  const t = tex(gl,w,h,internalFormat,format,type,filter); const fb_ = fbo(gl,t);
  return { w,h, tex:t, fbo:fb_ };
}

// ---------- Fragment shaders ----------
const commonUniforms = `
  uniform sampler2D uTex; // generic
  uniform sampler2D uVel; // velocity
  uniform sampler2D uSrc; // field source
  uniform sampler2D uP;   // pressure
  uniform sampler2D uDiv; // divergence
  uniform sampler2D uTemp;// temperature
  uniform sampler2D uPred;// predictor (MacCormack)
  uniform vec2 uTexel;    // 1/size
  uniform float uDt;      // delta time
`;

// MacCormack advection (forward predictor)
const fsAdvectFwd = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uDissipation;
  void main(){
    vec2 vel = texture(uVel, vUv).xy;
    vec2 coord = vUv - uDt * vel * uTexel; // backtrace
    frag = texture(uSrc, coord) * uDissipation;
  }
`;
// MacCormack corrector
const fsAdvectBwd = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uDissipation;
  void main(){
    vec2 vel = texture(uVel, vUv).xy;
    vec4 pred = texture(uPred, vUv);
    vec2 fwd = vUv + uDt * vel * uTexel; // forward sample of predictor
    vec4 predFwd = texture(uPred, fwd);
    vec4 src = texture(uSrc, vUv);
    vec4 corr = pred + 0.5*(src - predFwd);
    frag = clamp(corr, 0.0, 10.0) * uDissipation;
  }
`;

const fsDivergence = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  void main(){
    vec2 L = texture(uVel, vUv - vec2(uTexel.x,0.0)).xy;
    vec2 R = texture(uVel, vUv + vec2(uTexel.x,0.0)).xy;
    vec2 B = texture(uVel, vUv - vec2(0.0,uTexel.y)).xy;
    vec2 T = texture(uVel, vUv + vec2(0.0,uTexel.y)).xy;
    float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
    frag = vec4(div,0,0,1);
  }
`;

const fsJacobi = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uAlpha; uniform float uRBeta;
  void main(){
    float L = texture(uP, vUv - vec2(uTexel.x,0.0)).x;
    float R = texture(uP, vUv + vec2(uTexel.x,0.0)).x;
    float B = texture(uP, vUv - vec2(0.0,uTexel.y)).x;
    float T = texture(uP, vUv + vec2(0.0,uTexel.y)).x;
    float D = texture(uDiv, vUv).x;
    float P = (L+R+B+T + uAlpha * D) * uRBeta;
    frag = vec4(P,0,0,1);
  }
`;

const fsGradientSub = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  void main(){
    float L = texture(uP, vUv - vec2(uTexel.x,0.0)).x;
    float R = texture(uP, vUv + vec2(uTexel.x,0.0)).x;
    float B = texture(uP, vUv - vec2(0.0,uTexel.y)).x;
    float T = texture(uP, vUv + vec2(0.0,uTexel.y)).x;
    vec2 vel = texture(uVel, vUv).xy - 0.5 * vec2(R - L, T - B);
    frag = vec4(vel,0,1);
  }
`;

const fsCurl = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  void main(){
    float L = texture(uVel, vUv - vec2(uTexel.x,0.0)).y;
    float R = texture(uVel, vUv + vec2(uTexel.x,0.0)).y;
    float B = texture(uVel, vUv - vec2(0.0,uTexel.y)).x;
    float T = texture(uVel, vUv + vec2(0.0,uTexel.y)).x;
    float curl = (R - L) - (T - B);
    frag = vec4(curl,0,0,1);
  }
`;

const fsVorticity = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform sampler2D uCurl;
  uniform float uEps;
  void main(){
    float cL = abs(texture(uCurl, vUv - vec2(uTexel.x,0.0)).x);
    float cR = abs(texture(uCurl, vUv + vec2(uTexel.x,0.0)).x);
    float cB = abs(texture(uCurl, vUv - vec2(0.0,uTexel.y)).x);
    float cT = abs(texture(uCurl, vUv + vec2(0.0,uTexel.y)).x);
    vec2 grad = 0.5 * vec2(cR - cL, cT - cB);
    float curl = texture(uCurl, vUv).x;
    vec2 N = normalize(grad + vec2(1e-5));
    vec2 F = uEps * curl * vec2(N.y, -N.x);
    vec2 vel = texture(uVel, vUv).xy + F * uDt;
    frag = vec4(vel,0,1);
  }
`;

const fsBuoyancy = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uSigma; // temp weight
  uniform float uKappa; // density weight
  uniform float uScale;
  void main(){
    float T = texture(uTemp, vUv).x;
    float d = texture(uTex, vUv).w;
    vec2 vel = texture(uVel, vUv).xy;
    vel.y += (uSigma * T - uKappa * d) * uDt * uScale;
    frag = vec4(vel,0,1);
  }
`;

const fsSplat = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform vec2 uPoint;  uniform vec4 uColor;  uniform float uRadius;
  void main(){
    float d = distance(vUv, uPoint);
    float a = exp(-d*d / uRadius);
    vec4 base = texture(uSrc, vUv);
    frag = base + uColor * a;
  }
`;

const fsWind = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uStrength; uniform float uTime;
  void main(){
    float k1 = 3.0, k2 = 5.0;
    float s = sin((vUv.y*k1 + uTime*0.7))*cos((vUv.x*k2 - uTime*0.5));
    float c = cos((vUv.x*k1 - uTime*0.6))*sin((vUv.y*k2 + uTime*0.4));
    vec2 w = vec2(c, s) * uStrength;
    vec2 vel = texture(uVel, vUv).xy + w * uDt;
    frag = vec4(vel,0,1);
  }
`;

// Fluid: viscosity blur (small kernel)
const fsViscosity = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uVisc;
  void main(){
    vec2 t = uTexel;
    vec2 v0 = texture(uVel, vUv).xy;
    vec2 v1 = texture(uVel, vUv + vec2(+t.x,0.0)).xy;
    vec2 v2 = texture(uVel, vUv + vec2(-t.x,0.0)).xy;
    vec2 v3 = texture(uVel, vUv + vec2(0.0,+t.y)).xy;
    vec2 v4 = texture(uVel, vUv + vec2(0.0,-t.y)).xy;
    vec2 blur = (v0*4.0 + v1 + v2 + v3 + v4) / 8.0;
    vec2 outV = mix(v0, blur, clamp(uVisc, 0.0, 1.0));
    frag = vec4(outV,0,1);
  }
`;
// Fluid: curvature force from dye alpha
const fsCurvatureForce = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uStrength;
  void main(){
    float aC = texture(uTex, vUv).w;
    float aL = texture(uTex, vUv - vec2(uTexel.x,0.0)).w;
    float aR = texture(uTex, vUv + vec2(uTexel.x,0.0)).w;
    float aB = texture(uTex, vUv - vec2(0.0,uTexel.y)).w;
    float aT = texture(uTex, vUv + vec2(0.0,uTexel.y)).w;
    vec2 grad = 0.5 * vec2(aR - aL, aT - aB);
    vec2 vel = texture(uVel, vUv).xy - uStrength * grad * uDt;
    frag = vec4(vel,0,1);
  }
`;

// Smoke: ambient temperature gradient & evaporation near top
const fsAmbientTemp = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uGrad; // gradient strength
  void main(){
    float T = texture(uTemp, vUv).x + uGrad * (vUv.y - 0.5);
    frag = vec4(T,0,0,1);
  }
`;
const fsEvaporate = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uEdge; // top edge fade start (0.6..0.95)
  uniform float uAmount; // fade amount
  void main(){
    vec4 dye = texture(uSrc, vUv);
    float k = smoothstep(uEdge, 1.0, vUv.y) * uAmount;
    dye *= (1.0 - k);
    frag = dye;
  }
`;

// Plasma: vector potential A update and Lorentz-like force
const fsAUpdate = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uDamp; // eta
  void main(){
    float L = texture(uVel, vUv - vec2(uTexel.x,0.0)).y;
    float Rv= texture(uVel, vUv + vec2(uTexel.x,0.0)).y;
    float B = texture(uVel, vUv - vec2(0.0,uTexel.y)).x;
    float T = texture(uVel, vUv + vec2(0.0,uTexel.y)).x;
    float curlZ = (Rv - L) - (T - B);
    float Az = texture(uSrc, vUv).x;
    Az = Az + uDt * (curlZ - uDamp * Az);
    frag = vec4(Az,0.0,0.0,1.0);
  }
`;
const fsLorentz = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform float uScale;
  void main(){
    float AzL = texture(uTex, vUv - vec2(uTexel.x,0.0)).x;
    float AzR = texture(uTex, vUv + vec2(uTexel.x,0.0)).x;
    float AzB = texture(uTex, vUv - vec2(0.0,uTexel.y)).x;
    float AzT = texture(uTex, vUv + vec2(0.0,uTexel.y)).x;
    vec2 B = vec2( (AzT - AzB)*0.5, -(AzR - AzL)*0.5 );
    float ByL = (texture(uTex, vUv - vec2(uTexel.x,0.0)).xy).x;
    float ByR = (texture(uTex, vUv + vec2(uTexel.x,0.0)).xy).x;
    float BxB = (texture(uTex, vUv - vec2(0.0,uTexel.y)).xy).y;
    float BxT = (texture(uTex, vUv + vec2(0.0,uTexel.y)).xy).y;
    float Jz = 0.5 * ((ByR - ByL) - (BxT - BxB));
    vec2 F = Jz * vec2(-B.y, B.x) * uScale;
    vec2 vel = texture(uVel, vUv).xy + F * uDt;
    frag = vec4(vel,0,1);
  }
`;

// Stats: velocity magnitude
const fsVelMag = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  void main(){
    vec2 v = texture(uVel, vUv).xy;
    float m = length(v);
    frag = vec4(m,0,0,1);
  }
`;

// Composite (in PART 2)
const fsComposite = `
  in vec2 vUv; out vec4 frag; ${commonUniforms}
  uniform sampler2D uDye;   // RGB=dye, A=density
  uniform vec3 uGlowColor;  // plasma glow tint
  uniform float uExposure;  uniform float uGamma;  uniform float uGlow;
  void main(){
    vec4 dye = texture(uDye, vUv);
    float t = texture(uTemp, vUv).x;
    vec3 base = clamp(dye.rgb, 0.0, 10.0) * clamp(dye.a, 0.0, 1.5);
    vec3 mapped = 1.0 - exp(-base * max(uExposure, 0.0001));
    float glow = smoothstep(0.35, 0.95, t);
    mapped += uGlowColor * glow * uGlow;
    mapped = pow(clamp(mapped, 0.0, 1.0), vec3(1.0/max(uGamma, 0.0001)));
    frag = vec4(mapped, 1.0);
  }
`;

// ---------- React component (PART 1 up to runDiagnostics) ----------
export default function ElementalFieldLab(){
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const glRef = useRef(null);
  const simRef = useRef(null);
  const [playing, setPlaying] = useState(true);
  const [overlay, setOverlay] = useState("");
  const [fps, setFps] = useState(0);
  const statsRef = useRef({ maxVel:0, avgDivPre:0, avgDivPost:0, substeps:1, nans:0 });

  const [ui, setUI] = useState({
    resScale: 1.0,
    pressureIters: 24,
    vorticity: 10.0,
    buoyancySigma: 1.2,
    buoyancyKappa: 0.2,
    buoyancyScale: 1.0,
    wind: 0.6,
    dyeDissipation: 0.994,
    velDissipation: 0.999,
    exposure: 1.1,
    gamma: 2.0,
    glow: 0.9,
    emission: 'fluid',
    color: '#8fd3ff',
  });

  const emittersRef = useRef([
    { x:0.25, y:0.5, color:[0.56,0.83,1.0,0.9], type:'smoke' },
    { x:0.75, y:0.5, color:[1.0,0.45,0.7,0.9], type:'fluid' },
    { x:0.5, y:0.25, color:[1.0,0.95,0.5,0.9], type:'plasma' }
  ]);

  useEffect(()=>{
    const canvas = canvasRef.current; if (!canvas) return;
    let gl; try { gl = makeGL(canvas); } catch (e){ console.error(e); setOverlay(String(e.message||e)); return; }
    glRef.current = gl;

    // Build programs
    let prog = {};
    try {
      prog.quad = makeQuad(gl);
      prog.advF = makeProgram(gl, fsAdvectFwd, '', 'advectFwd');
      prog.advB = makeProgram(gl, fsAdvectBwd, '', 'advectBwd');
      prog.divg = makeProgram(gl, fsDivergence, '', 'divergence');
      prog.jac = makeProgram(gl, fsJacobi, '', 'jacobi');
      prog.grad = makeProgram(gl, fsGradientSub, '', 'gradient');
      prog.curl = makeProgram(gl, fsCurl, '', 'curl');
      prog.vort = makeProgram(gl, fsVorticity, '', 'vorticity');
      prog.buoy = makeProgram(gl, fsBuoyancy, '', 'buoyancy');
      prog.splat = makeProgram(gl, fsSplat, '', 'splat');
      prog.wind = makeProgram(gl, fsWind, '', 'wind');
      prog.visc = makeProgram(gl, fsViscosity, '', 'viscosity');
      prog.curv = makeProgram(gl, fsCurvatureForce, '', 'curvature');
      prog.atmp = makeProgram(gl, fsAmbientTemp, '', 'ambientTemp');
      prog.evap = makeProgram(gl, fsEvaporate, '', 'evaporate');
      prog.aupd = makeProgram(gl, fsAUpdate, '', 'Aupdate');
      prog.lorz = makeProgram(gl, fsLorentz, '', 'Lorentz');
      prog.vmag = makeProgram(gl, fsVelMag, '', 'velMag');
      prog.comp = makeProgram(gl, fsComposite, '', 'composite');
    } catch (e) {
      console.error('[Pipeline Build Failed]', e); setOverlay(String(e.message||e)); return;
    }

    const sim = { gl, prog, t:0, w:0, h:0 };
    simRef.current = sim;

    selfTest(sim, setOverlay);

    const fit = ()=>{
      const dpr = Math.min(2, window.devicePixelRatio||1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(32, Math.floor(rect.width * dpr));
      canvas.height = Math.max(32, Math.floor(rect.height * dpr));
      sim.w = Math.floor(canvas.width * ui.resScale);
      sim.h = Math.floor(canvas.height * ui.resScale);
      initTargets(sim);
    };
    const ro = new ResizeObserver(fit); ro.observe(canvas); fit();

    let last = performance.now(); let acc = 0, frames = 0;
    const loop = ()=>{
      rafRef.current = requestAnimationFrame(loop);
      const now = performance.now(); const dt = Math.min(0.033, Math.max(0.008, (now-last)/1000)); last = now; acc += dt; frames++;
      if (acc>0.5){ setFps(Math.round(frames/acc)); acc=0; frames=0; }
      if (playing) step(sim, ui, dt, emittersRef.current, statsRef.current);
      draw(sim, ui, statsRef.current);
    };
    loop();

    const onPointer = (e)=>{
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left)/rect.width; const y = 1-((e.clientY - rect.top)/rect.height);
      if (e.buttons===1){
        const col = hexToRgba(ui.color, ui.emission==='smoke'?0.8:0.6);
        emittersRef.current.push({ x, y, color: col, type: ui.emission });
        if (emittersRef.current.length>12) emittersRef.current.shift();
        splatAt(sim, x, y, ui.emission, col);
      } else if (e.type==='pointerdown') {
        const col = hexToRgba(ui.color, ui.emission==='smoke'?0.8:0.6);
        splatAt(sim, x, y, ui.emission, col);
      }
    };
    canvas.addEventListener('pointerdown', onPointer);
    canvas.addEventListener('pointermove', onPointer);

    return ()=>{ cancelAnimationFrame(rafRef.current); ro.disconnect(); canvas.removeEventListener('pointerdown', onPointer); canvas.removeEventListener('pointermove', onPointer); dispose(sim); };
  }, [playing, ui.resScale]);

  const runDiagnostics = () => {
    try{
      const sim = simRef.current; const gl = glRef.current;
      if (!gl || !sim) { alert('WebGL2 context not initialized'); return; }
      const msgs = [];
      msgs.push('WebGL2: OK');
      const ext = gl.getExtension('EXT_color_buffer_float'); msgs.push('EXT_color_buffer_float: ' + (ext ? 'OK' : 'MISSING'));
      const okFbo = (buf)=>{ if(!buf) return false; const f1=buf.fbo||buf.fboR; const f2=buf.fboW||buf.fbo; gl.bindFramebuffer(gl.FRAMEBUFFER, f1); const s1 = gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE; gl.bindFramebuffer(gl.FRAMEBUFFER, f2); const s2 = gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE; gl.bindFramebuffer(gl.FRAMEBUFFER, null); return s1 && s2; };
      ['vel','dye','temp','pressure','div','curl','tmpVel','tmpDye','tmpTemp','A'].forEach(k=> msgs.push(`${k}: ${okFbo(sim[k])?'OK':'BAD'}`));
      alert(msgs.join('\n'));
    } catch(e){ alert('Diagnostics error: '+e.message); }
  };

  return (
    <div className="w-full h-full grid grid-rows-[48px_1fr_52px] bg-neutral-950 text-neutral-100">
      {/* top bar */}
      <div className="flex items-center gap-3 px-3 border-b border-neutral-800 bg-neutral-900/70">
        <button
          className="px-2 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-xs"
          onClick={()=>setPlaying(p=>!p)}
        >
          {playing? 'Pause':'Run'}
        </button>

        <button
          className="px-2 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-xs"
          onClick={()=>{ const sim=simRef.current; if(sim) clearSim(sim); }}
        >
          Reset
        </button>

        <button
          className="px-2 py-1 rounded bg-sky-700 hover:bg-sky-600 text-xs"
          onClick={runDiagnostics}
        >
          Diagnostics
        </button>

        <label className="text-xs flex items-center gap-2">Resolution
          <select
            className="bg-neutral-800 rounded px-2 py-1"
            value={ui.resScale}
            onChange={(e)=>setUI({...ui, resScale: parseFloat(e.target.value)})}
          >
            <option value={0.5}>0.5×</option>
            <option value={1.0}>1.0×</option>
            <option value={1.5}>1.5×</option>
          </select>
        </label>

        <label className="text-xs flex items-center gap-2">Pressure Iters
          <input
            type="range" min={8} max={80} step={1}
            value={ui.pressureIters}
            onChange={(e)=>setUI({...ui, pressureIters: parseInt(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Vorticity
          <input
            type="range" min={0} max={30} step={0.5}
            value={ui.vorticity}
            onChange={(e)=>setUI({...ui, vorticity: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Buoyancy
          <input
            type="range" min={0} max={3} step={0.1}
            value={ui.buoyancySigma}
            onChange={(e)=>setUI({...ui, buoyancySigma: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Density
          <input
            type="range" min={-1.0} max={1.0} step={0.05}
            value={ui.buoyancyKappa}
            onChange={(e)=>setUI({...ui, buoyancyKappa: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Wind
          <input
            type="range" min={0} max={3} step={0.05}
            value={ui.wind}
            onChange={(e)=>setUI({...ui, wind: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Exposure
          <input
            type="range" min={0.2} max={3.0} step={0.05}
            value={ui.exposure}
            onChange={(e)=>setUI({...ui, exposure: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Gamma
          <input
            type="range" min={1.0} max={2.6} step={0.05}
            value={ui.gamma}
            onChange={(e)=>setUI({...ui, gamma: parseFloat(e.target.value)})}
          />
        </label>

        <label className="text-xs flex items-center gap-2">Glow
          <input
            type="range" min={0.0} max={2.0} step={0.05}
            value={ui.glow}
            onChange={(e)=>setUI({...ui, glow: parseFloat(e.target.value)})}
          />
        </label>

        <div className="ml-auto flex items-center gap-3">
          <label className="text-xs flex items-center gap-2">Emit
            <select
              className="bg-neutral-800 rounded px-2 py-1"
              value={ui.emission}
              onChange={(e)=>setUI({...ui, emission:e.target.value})}
            >
              <option value="fluid">Fluid</option>
              <option value="smoke">Smoke</option>
              <option value="plasma">Plasma</option>
            </select>
          </label>
          <input
            aria-label="Color"
            type="color"
            value={ui.color}
            onChange={(e)=>setUI({...ui, color:e.target.value})}
          />
          <span className="opacity-70 text-xs">{fps} fps</span>
        </div>
      </div>

      {/* canvas */}
      <div className="relative">
        <canvas ref={canvasRef} className="w-full h-full block" />
        {overlay && (
          <div className="absolute inset-x-0 top-2 mx-auto max-w-[80ch] bg-red-900/70 text-red-100 text-xs rounded p-2 border border-red-700 whitespace-pre-wrap">
            <strong className="block mb-1">Shader/Pipeline Error</strong>
            {overlay}
          </div>
        )}
        {/* Dev HUD */}
        <div className="pointer-events-none absolute top-2 left-3 text-[11px] leading-4 bg-neutral-900/60 px-2 py-1 rounded border border-neutral-700">
          <div>
            max|v|: {statsRef.current.maxVel.toFixed(3)}  •  ⟨|div|⟩ pre: {statsRef.current.avgDivPre.toFixed(4)}  post: {statsRef.current.avgDivPost.toFixed(4)}
          </div>
          <div>
            substeps: {statsRef.current.substeps}  •  NaNs: {statsRef.current.nans}  •  fps: {fps}
          </div>
        </div>
        <div className="pointer-events-none absolute bottom-2 left-3 text-xs opacity-60">
          Drag to paint velocity + dye. Click to burst. Add multiple types to mix.
        </div>
      </div>

      {/* bottom bar */}
      <div className="flex items-center gap-4 px-3 border-t border-neutral-800 bg-neutral-900/70 text-xs">
        <div>Elemental Field Lab — Phase 0–2 Engine</div>
        <div className="opacity-70">• MacCormack • CFL Substeps • Fluid/Smoke/Plasma Plugins • Dev HUD</div>
      </div>
    </div>
  );
}

// ---------- Simulation buffers & steps ----------
function initTargets(sim){
  const gl = sim.gl; const w = sim.w, h = sim.h; if (!gl || !w || !h) return;
  dispose(sim);
  const LINEAR = gl.LINEAR; const NEAREST = gl.NEAREST; const HALF = gl.HALF_FLOAT || 0x140B;

  sim.vel    = doubleFBO(gl,w,h, gl.RG16F,   gl.RG,   HALF, LINEAR);
  sim.tmpVel = doubleFBO(gl,w,h, gl.RG16F,   gl.RG,   HALF, LINEAR);
  sim.dye    = doubleFBO(gl,w,h, gl.RGBA16F, gl.RGBA, HALF, LINEAR);
  sim.tmpDye = doubleFBO(gl,w,h, gl.RGBA16F, gl.RGBA, HALF, LINEAR);
  sim.temp   = doubleFBO(gl,w,h, gl.R16F,    gl.RED,  HALF, LINEAR);
  sim.tmpTemp= doubleFBO(gl,w,h, gl.R16F,    gl.RED,  HALF, LINEAR);

  sim.pressure = doubleFBO(gl,w,h, gl.R16F, gl.RED, HALF, NEAREST);
  sim.div      = doubleFBO(gl,w,h, gl.R16F, gl.RED, HALF, NEAREST);
  sim.curl     = doubleFBO(gl,w,h, gl.R16F, gl.RED, HALF, NEAREST);

  // Plasma vector potential A (Az only stored in R)
  sim.A = doubleFBO(gl,w,h, gl.R16F, gl.RED, HALF, LINEAR);

  // Stats downsample buffers
  const s = 64;
  sim.statsVel = singleFBO(gl, s, s, gl.R16F, gl.RED, HALF, NEAREST);
  sim.statsDiv = singleFBO(gl, s, s, gl.R16F, gl.RED, HALF, NEAREST);

  clearSim(sim);
}

function dispose(sim){
  const gl = sim?.gl; if (!gl) return;
  const kill = (buf)=>{
    if(!buf) return;
    if(buf.read){
      gl.deleteTexture(buf.read); gl.deleteTexture(buf.write);
      gl.deleteFramebuffer(buf.fboR); gl.deleteFramebuffer(buf.fboW);
    } else {
      gl.deleteTexture(buf.tex); gl.deleteFramebuffer(buf.fbo);
    }
  };
  ['vel','tmpVel','dye','tmpDye','temp','tmpTemp','pressure','div','curl','A','statsVel','statsDiv'].forEach(k=> kill(sim[k]));
}

function clearSim(sim){
  const { gl, w, h } = sim; if (!gl || !w || !h) return;
  const clearDouble = (buf)=>{
    if(!buf) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, buf.fboR); gl.viewport(0,0,w,h);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, buf.fboW); gl.clear(gl.COLOR_BUFFER_BIT);
  };
  const clearSingle = (buf)=>{
    if(!buf) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, buf.fbo); gl.viewport(0,0,buf.w,buf.h);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
  };
  ['vel','tmpVel','dye','tmpDye','temp','tmpTemp','pressure','div','curl','A'].forEach(k=> clearDouble(sim[k]));
  clearSingle(sim.statsVel); clearSingle(sim.statsDiv);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function isProgramOk(gl, p){
  if (!p) return false;
  try { return !!gl.isProgram && gl.isProgram(p) === true; } catch { return false; }
}

// MacCormack advect helper (field = doubleFBO, srcFormat agnostic)
function advectMC(sim, field, vel, diss){
  const texel = [1/sim.w, 1/sim.h];

  // forward predictor -> tmp
  run(sim, sim.prog.advF, (set)=>{
    set('uVel', vel.read, 1);
    set('uSrc', field.read, 0);
    set('uTexel', texel);
    set('uDt', sim._dtSub);
    set('uDissipation', diss);
  }, field===sim.vel? sim.tmpVel.fboW : field===sim.temp? sim.tmpTemp.fboW : sim.tmpDye.fboW);

  if (field===sim.vel) sim.tmpVel.swap(); else if(field===sim.temp) sim.tmpTemp.swap(); else sim.tmpDye.swap();
  const predTex = field===sim.vel? sim.tmpVel.read : field===sim.temp? sim.tmpTemp.read : sim.tmpDye.read;

  // backward corrector -> field
  run(sim, sim.prog.advB, (set)=>{
    set('uVel', vel.read, 1);
    set('uSrc', field.read, 0);
    set('uPred', predTex, 2);
    set('uTexel', texel);
    set('uDt', sim._dtSub);
    set('uDissipation', diss);
  }, field.fboW);
  field.swap();
}

function computeStats(sim, stats){
  const gl = sim.gl; const texel = [1/sim.w,1/sim.h];

  // vel magnitude -> statsVel
  run(sim, sim.prog.vmag, (set)=>{
    set('uVel', sim.vel.read, 1);
    set('uTexel', texel);
    set('uDt', 0.016);
  }, sim.statsVel.fbo);

  // divergence (pre-pressure) using current vel -> statsDiv
  run(sim, sim.prog.divg, (set)=>{
    set('uVel', sim.vel.read, 1);
    set('uTexel', texel);
  }, sim.statsDiv.fbo);

  // read back
  const s = sim.statsVel.w; const n = s*s;
  const buf = new Float32Array(n*4);
  const buf2 = new Float32Array(n*4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, sim.statsVel.fbo);
  gl.readPixels(0,0,s,s, gl.RGBA, gl.FLOAT, buf);
  gl.bindFramebuffer(gl.FRAMEBUFFER, sim.statsDiv.fbo);
  gl.readPixels(0,0,s,s, gl.RGBA, gl.FLOAT, buf2);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  let maxV=0, nans=0, sumDiv=0, cnt=0;
  for (let i=0;i<buf.length;i+=4){
    const v=buf[i];
    if(!Number.isFinite(v)) { nans++; continue; }
    if(v>maxV) maxV=v;
  }
  for (let i=0;i<buf2.length;i+=4){
    const d=Math.abs(buf2[i]);
    if(Number.isFinite(d)){ sumDiv+=d; cnt++; } else { nans++; }
  }
  stats.maxVel = maxV;
  stats.avgDivPre = cnt? (sumDiv/cnt):0;
  stats.nans = nans;
}

function step(sim, ui, dt, emitters, stats){
  const gl = sim.gl; const texel = [1/sim.w, 1/sim.h]; sim.t += dt;
  gl.viewport(0,0,sim.w,sim.h); gl.disable(gl.BLEND);

  // Element coefficients
  const K = coeffFor(ui.emission);

  // 0) Wind (pre-stats)
  run(sim, sim.prog.wind, (set)=>{
    set('uVel', sim.vel.read, 1);
    set('uTexel', texel);
    set('uDt', dt);
    set('uStrength', ui.wind * K.windScale);
    set('uTime', sim.t);
  }, sim.vel.fboW); sim.vel.swap();

  // Phase 0: compute stats
  computeStats(sim, stats);

  // CFL substeps
  const cell = Math.max(texel[0], texel[1]);
  const cflTarget = 0.6;
  let sub = Math.max(1, Math.min(4, Math.ceil((stats.maxVel * dt / cell) / cflTarget)));
  stats.substeps = sub; sim._dtSub = dt / sub;

  for (let s=0; s<sub; s++){
    // 1) Advect velocity (MacCormack)
    advectMC(sim, sim.vel, sim.vel, clamp(ui.velDissipation * K.velDissip, 0.90, 1.0));

    // 2) Curl & Vorticity
    run(sim, sim.prog.curl, (set)=>{ set('uVel', sim.vel.read, 1); set('uTexel', texel); }, sim.curl.fboW); sim.curl.swap();
    run(sim, sim.prog.vort, (set)=>{
      set('uVel', sim.vel.read, 1); set('uCurl', sim.curl.read, 3);
      set('uTexel', texel); set('uDt', sim._dtSub);
      set('uEps', ui.vorticity * K.vorticityScale);
    }, sim.vel.fboW); sim.vel.swap();

    // 3) Divergence
    run(sim, sim.prog.divg, (set)=>{ set('uVel', sim.vel.read, 1); set('uTexel', texel); }, sim.div.fboW); sim.div.swap();

    // 4) Pressure solve
    const iters = Math.max(8, Math.floor(ui.pressureIters / sub));
    for (let i=0;i<iters;i++){
      run(sim, sim.prog.jac, (set)=>{
        set('uP', sim.pressure.read, 4);
        set('uDiv', sim.div.read, 5);
        set('uTexel', texel); set('uAlpha', -1.0); set('uRBeta', 0.25);
      }, sim.pressure.fboW);
      sim.pressure.swap();
    }

    // 5) Gradient subtract
    run(sim, sim.prog.grad, (set)=>{ set('uVel', sim.vel.read, 1); set('uP', sim.pressure.read, 4); set('uTexel', texel); }, sim.vel.fboW); sim.vel.swap();

    // 6) Element plugin forces
    if (ui.emission==='fluid'){
      // viscosity
      run(sim, sim.prog.visc, (set)=>{ set('uVel', sim.vel.read, 1); set('uTexel', texel); set('uDt', sim._dtSub); set('uVisc', 0.25); }, sim.vel.fboW); sim.vel.swap();
      // curvature from dye alpha
      run(sim, sim.prog.curv, (set)=>{ set('uVel', sim.vel.read, 1); set('uTex', sim.dye.read, 0); set('uTexel', texel); set('uDt', sim._dtSub); set('uStrength', 0.15); }, sim.vel.fboW); sim.vel.swap();
    } else if (ui.emission==='smoke'){
      // ambient temperature gradient
      run(sim, sim.prog.atmp, (set)=>{ set('uTemp', sim.temp.read, 6); set('uSrc', sim.temp.read, 0); set('uTexel', texel); set('uDt', sim._dtSub); set('uGrad', 0.25); }, sim.temp.fboW); sim.temp.swap();
      // top evaporation on dye
      run(sim, sim.prog.evap, (set)=>{ set('uSrc', sim.dye.read, 0); set('uTexel', texel); set('uDt', sim._dtSub); set('uEdge', 0.75); set('uAmount', 0.15); }, sim.dye.fboW); sim.dye.swap();
    } else if (ui.emission==='plasma'){
      // update A and apply Lorentz-like force
      run(sim, sim.prog.aupd, (set)=>{ set('uSrc', sim.A.read, 0); set('uVel', sim.vel.read, 1); set('uTexel', texel); set('uDt', sim._dtSub); set('uDamp', 0.6); }, sim.A.fboW); sim.A.swap();
      run(sim, sim.prog.lorz, (set)=>{ set('uVel', sim.vel.read, 1); set('uTex', sim.A.read, 0); set('uTexel', texel); set('uDt', sim._dtSub); set('uScale', 0.4); }, sim.vel.fboW); sim.vel.swap();
      // optional re-projection could be added here
    }

    // 7) Advect dye & temp (MacCormack)
    advectMC(sim, sim.dye, sim.vel,  clamp(ui.dyeDissipation * K.dyeDissip, 0.90, 1.0));
    advectMC(sim, sim.temp, sim.vel, clamp(0.998 * K.tempDissip,           0.90, 1.0));

    // 8) Buoyancy
    run(sim, sim.prog.buoy, (set)=>{
      set('uVel', sim.vel.read, 1); set('uTex', sim.dye.read, 0);
      set('uTemp', sim.temp.read, 6); set('uTexel', texel); set('uDt', sim._dtSub);
      set('uSigma', ui.buoyancySigma * K.buoySigma);
      set('uKappa', ui.buoyancyKappa * K.buoyKappa);
      set('uScale', K.buoyScale);
    }, sim.vel.fboW); sim.vel.swap();

    // 9) Emitters
    for (const e of emitters){ splatAt(sim, e.x, e.y, e.type, e.color, 0.012, K); }
  }

  // Phase 0: compute post-pressure avg|div| for HUD (recompute divergence)
  run(sim, sim.prog.divg, (set)=>{ set('uVel', sim.vel.read, 1); set('uTexel', texel); }, sim.statsDiv.fbo);
  const s2 = sim.statsDiv.w; const buf = new Float32Array(s2*s2*4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, sim.statsDiv.fbo);
  gl.readPixels(0,0,s2,s2, gl.RGBA, gl.FLOAT, buf);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  let sum=0, cnt=0; for(let i=0;i<buf.length;i+=4){ const d=Math.abs(buf[i]); if(Number.isFinite(d)){ sum+=d; cnt++; } }
  stats.avgDivPost = cnt? (sum/cnt):0;
}

function run(sim, prog, setup, targetFbo){
  const gl = sim.gl;
  if (!isProgramOk(gl, prog)) { onceWarn('prog_warn_'+(prog?.__id||Math.random()), '[run] skipped invalid program'); return; }
  gl.useProgram(prog); gl.bindVertexArray(sim.prog.quad);
  if (targetFbo) gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo); else gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  const set = (name, val, unit)=>{
    const loc = gl.getUniformLocation(prog, name); if (loc==null) return;
    if (typeof val === 'number') gl.uniform1f(loc, val);
    else if (Array.isArray(val)) {
      if (val.length===2) gl.uniform2fv(loc,val);
      else if(val.length===3) gl.uniform3fv(loc,val);
      else if(val.length===4) gl.uniform4fv(loc,val);
    }
    else if (val && val.__TEXTURE__) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, val.__TEXTURE__); gl.uniform1i(loc, unit);
    } else if (val instanceof WebGLTexture) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, val); gl.uniform1i(loc, unit);
    }
  };

  setup((name, val, unit)=>{
    // wrap known ping-pong textures
    if (val === sim.vel.read)      val = wrapTex(gl, sim.vel.read);
    else if (val === sim.dye.read) val = wrapTex(gl, sim.dye.read);
    else if (val === sim.temp.read)val = wrapTex(gl, sim.temp.read);
    else if (val === sim.pressure.read) val = wrapTex(gl, sim.pressure.read);
    else if (val === sim.div.read)  val = wrapTex(gl, sim.div.read);
    else if (val === sim.curl.read) val = wrapTex(gl, sim.curl.read);
    else if (val === sim.A.read)    val = wrapTex(gl, sim.A.read);
    set(name, val, unit);
  });

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function wrapTex(gl, t){ return { __TEXTURE__: t, gl }; }

function draw(sim, ui){
  const gl = sim.gl; gl.viewport(0,0,sim.w,sim.h);
  run(sim, sim.prog.comp, (set)=>{
    set('uDye',  wrapTex(gl, sim.dye.read), 7);
    set('uTemp', wrapTex(gl, sim.temp.read), 8);
    set('uGlowColor', hexToRgb(ui.color));
    set('uTexel', [1/sim.w, 1/sim.h]);
    set('uDt', 0.016);
    set('uExposure', ui.exposure);
    set('uGamma', ui.gamma);
    set('uGlow', ui.glow);
  }, null);
}

function splatAt(sim, x, y, type, color, radius=0.02, K={}){
  const gl = sim.gl; const texel=[1/sim.w,1/sim.h];
  const speed = 0.6 * (K.splatVel||1.0);
  const ang = Math.random()*Math.PI*2; const v = [Math.cos(ang)*speed, Math.sin(ang)*speed];
  const r = clamp(radius * (K.splatRadius||1.0), 0.003, 0.08);

  // velocity impulse
  run(sim, sim.prog.splat, (set)=>{
    set('uSrc', wrapTex(gl, sim.vel.read), 0);
    set('uVel', wrapTex(gl, sim.vel.read), 1);
    set('uTexel', texel); set('uDt', 0.016);
    set('uPoint', [x,y]); set('uColor', [v[0], v[1], 0, 0]); set('uRadius', r);
  }, sim.vel.fboW); sim.vel.swap();

  // plasma temperature
  if (type==='plasma'){
    run(sim, sim.prog.splat, (set)=>{
      set('uSrc', wrapTex(gl, sim.temp.read), 0);
      set('uVel', wrapTex(gl, sim.vel.read), 1);
      set('uTexel', texel); set('uDt', 0.016);
      set('uPoint', [x,y]); set('uColor', [1.2,0,0,0]); set('uRadius', r*1.05);
    }, sim.temp.fboW); sim.temp.swap();
  }

  // dye injection
  run(sim, sim.prog.splat, (set)=>{
    set('uSrc', wrapTex(gl, sim.dye.read), 0);
    set('uVel', wrapTex(gl, sim.vel.read), 1);
    set('uTexel', texel); set('uDt', 0.016);
    set('uPoint', [x,y]); set('uColor', [color[0], color[1], color[2], clamp(color[3], 0.2, 1.1)]); set('uRadius', r);
  }, sim.dye.fboW); sim.dye.swap();
}

function hexToRgb(hex){
  const h = hex.replace('#','');
  const v = parseInt(h.length===3 ? h.split('').map(c=>c+c).join('') : h,16);
  return [((v>>16)&255)/255, ((v>>8)&255)/255, (v&255)/255];
}
function hexToRgba(hex, a=0.8){ const [r,g,b] = hexToRgb(hex); return [r,g,b,a]; }

// Element behavior presets for substepping-era engine
function coeffFor(kind){
  switch(kind){
    case 'fluid':  return { dyeDissip: 0.996, tempDissip: 0.999, velDissip: 1.000, buoySigma: 0.3, buoyKappa: 0.6, buoyScale: 1.0, vorticityScale: 0.9, windScale: 0.8, splatVel: 0.8, splatRadius: 1.0 };
    case 'smoke':  return { dyeDissip: 0.990, tempDissip: 0.998, velDissip: 0.998, buoySigma: 1.6, buoyKappa: -0.2, buoyScale: 1.2, vorticityScale: 1.3, windScale: 1.1, splatVel: 0.7, splatRadius: 1.1 };
    case 'plasma': return { dyeDissip: 0.995, tempDissip: 0.997, velDissip: 0.999, buoySigma: 2.0, buoyKappa: 0.1,  buoyScale: 1.4, vorticityScale: 1.6, windScale: 1.2, splatVel: 1.1, splatRadius: 0.9 };
    default:       return { dyeDissip: 0.994, tempDissip: 0.998, velDissip: 0.999, buoySigma: 1.0, buoyKappa: 0.2,  buoyScale: 1.0, vorticityScale: 1.0, windScale: 1.0, splatVel: 1.0, splatRadius: 1.0 };
  }
}

// ---------- Minimal self-tests (runtime) ----------
function selfTest(sim, setOverlay){
  try{
    const gl = sim.gl; const p = sim.prog;
    const stages = ['advF','advB','divg','jac','grad','curl','vort','buoy','splat','wind','visc','curv','atmp','evap','aupd','lorz','vmag','comp'];
    for (const k of stages){
      const ok = isProgramOk(gl, p[k]);
      console.assert(ok, `Program ${k} valid`);
      if(!ok) throw new Error(`Stage '${k}' failed`);
    }
    clearSim(sim);
    const err = gl.getError(); console.assert(err === gl.NO_ERROR, 'No GL error after init');
  } catch(e){ console.warn('[Elemental Field Lab] Self test failed:', e); setOverlay(String(e.message||e)); }
}
