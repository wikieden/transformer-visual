const NS='http://www.w3.org/2000/svg';
const CAPTURE=location.search.includes('capture');
if(CAPTURE)document.documentElement.classList.add('cap');
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}
function txt(s,attrs,parent){const e=el('text',attrs,parent);e.textContent=s;return e}
function clear(g){while(g.firstChild)g.removeChild(g.firstChild)}
let _s=1;function seed(s){_s=(s>>>0)||1}
function rnd(){_s=_s+0x6D2B79F5|0;let t=Math.imul(_s^_s>>>15,1|_s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}
function gauss(){let u=0;while(!u)u=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*rnd())}
function $(id){return document.getElementById(id)}
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function setupPlayer(A,ms){
  const play=$('play'),step=$('step'),reset=$('reset');let t=null;
  const stop=()=>{clearInterval(t);t=null;play.textContent='▶ 播放'};
  play.onclick=()=>{if(t){stop();return}play.textContent='❚❚ 暂停';t=setInterval(()=>{if(!A.next())stop()},ms)};
  step.onclick=()=>{stop();A.next()};
  reset.onclick=()=>{stop();A.reset(Date.now())};
  A.stop=stop;A.reset(CAPTURE?A.captureSeed||7:Date.now());window.ANIM=A;
}
