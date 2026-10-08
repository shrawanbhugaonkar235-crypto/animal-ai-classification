const $=id=>document.getElementById(id);
let selected=null,detections=[];
document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>show(b.dataset.view)));
function show(v){document.querySelectorAll(".view").forEach(x=>x.classList.toggle("active",x.id===v));document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("active",x.dataset.view===v));$("nav").classList.remove("open");window.scrollTo({top:0,behavior:"smooth"})}
$("menu").onclick=()=>$("nav").classList.toggle("open");
$("browse").onclick=()=>$("file").click();
$("file").onchange=()=>{if($("file").files[0])setFile($("file").files[0])};
$("remove").onclick=reset;
$("newPhoto").onclick=reset;
$("analyze").onclick=analyze;
["search","category","confidence"].forEach(id=>$(id).addEventListener("input",renderList));
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("drag")};
$("drop").ondragleave=()=>$("drop").classList.remove("drag");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("drag");if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0])};
$("drop").onclick=e=>{if(["drop","H3","P"].includes(e.target.id)||e.target.tagName==="H3"||e.target.tagName==="P")$("file").click()};
function setFile(f){
  if(!["image/jpeg","image/png","image/webp"].includes(f.type))return fail("Please choose JPG, JPEG, PNG or WEBP.");
  selected=f;$("filename").textContent=`${f.name} · ${(f.size/1024/1024).toFixed(2)} MB`;
  $("preview").src=URL.createObjectURL(f);$("drop").classList.add("hidden");$("previewBox").classList.remove("hidden");$("results").classList.add("hidden");hideError()
}
function reset(){selected=null;$("file").value="";$("preview").removeAttribute("src");$("drop").classList.remove("hidden");$("previewBox").classList.add("hidden");$("results").classList.add("hidden");hideError()}
function fail(t){$("error").textContent=t;$("error").classList.remove("hidden")}
function hideError(){$("error").classList.add("hidden");$("error").textContent=""}
async function compressImage(file){
  const bmp=await createImageBitmap(file),maxSide=1600,scale=Math.min(1,maxSide/bmp.width,maxSide/bmp.height);
  const c=document.createElement("canvas");c.width=Math.max(1,Math.round(bmp.width*scale));c.height=Math.max(1,Math.round(bmp.height*scale));
  c.getContext("2d").drawImage(bmp,0,0,c.width,c.height);
  let q=.82,blob=await new Promise(r=>c.toBlob(r,"image/jpeg",q));
  while(blob&&blob.size>4*1024*1024&&q>.45){q-=.08;blob=await new Promise(r=>c.toBlob(r,"image/jpeg",q))}
  if(!blob||blob.size>4*1024*1024)throw Error("Please choose a smaller photo.");
  return new File([blob],"upload.jpg",{type:"image/jpeg"});
}
async function analyze(){
  if(!selected)return fail("Select a photo first.");
  $("loading").classList.remove("hidden");$("analyze").disabled=true;hideError();
  try{
    const file=await compressImage(selected),fd=new FormData();fd.append("file",file);
    const r=await fetch("/api/predict",{method:"POST",body:fd}),d=await r.json();
    if(!r.ok)throw Error(d.detail||"Prediction failed.");
    detections=d.detections||[]; $("results").classList.remove("hidden");
    $("total").textContent=d.total_detections;$("land").textContent=d.summary.land_animals;$("sea").textContent=d.summary.sea_animals;$("birds").textContent=d.summary.birds;$("avg").textContent=d.summary.average_confidence+"%";$("ptime").textContent=d.processing_time_ms+" ms";$("model").textContent=d.model_name;$("output").src=d.annotated_image+"?t="+Date.now();renderList()
  }catch(e){fail(e.message||"Could not analyze the image.")}finally{$("loading").classList.add("hidden");$("analyze").disabled=false}
}
function renderList(){
  const q=$("search").value.toLowerCase(),cat=$("category").value,min=Number($("confidence").value);
  const rows=detections.filter(x=>x.name.toLowerCase().includes(q)&&(cat==="All"||x.category===cat)&&x.confidence_percent>=min);
  $("list").innerHTML=rows.length?rows.map(x=>`<div class="item"><div class="itemtop"><div><b>${escapeHtml(x.name)}</b><small>${escapeHtml(x.category)} · ${escapeHtml(x.status)}</small></div><b>${x.confidence_percent}%</b></div><div class="prog"><span style="width:${Math.min(100,x.confidence_percent)}%"></span></div><div class="bbox">Box: [${x.bbox.x1}, ${x.bbox.y1}] → [${x.bbox.x2}, ${x.bbox.y2}]</div></div>`).join(""):'<div class="panel" style="box-shadow:none;text-align:center">No detections match the filters.</div>'
}
function escapeHtml(v){return String(v).replace(/[&<>"']/g,s=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[s]))}
async function load(){
  try{
    const h=await (await fetch("/api/health")).json();$("health").textContent=h.model_loaded?"Model ready · "+h.model_name:"Model unavailable";$("health").className="status "+(h.model_loaded?"ok":"bad");
    const c=await (await fetch("/api/classes")).json();const set=new Set(c.actual_model_classes.map(x=>x.toLowerCase().trim()));
    $("coverage").innerHTML=c.model_loaded?`<b>Loaded model:</b> ${escapeHtml(c.model_name)} · <b>${c.actual_model_classes.length}</b> actual classes. Classes not in the model are marked as training-required.`:"<b>AI model unavailable.</b> Add trained weights or configure a compatible model.";
    const intended=c.classes||[];const groups={landClasses:[],seaClasses:[],birdClasses:[]};
    for(const x of intended){const k=x.category==="Land Animals"?"landClasses":x.category==="Sea Animals"?"seaClasses":"birdClasses";groups[k].push(x)}
    for(const [k,a] of Object.entries(groups))$(k).innerHTML=a.map(x=>`<div class="chip ${x.supported?"supported":""}"><b>${escapeHtml(x.name)}</b><small>${x.supported?"✓ Supported":"○ Training required"}</small></div>`).join("")
  }catch{$("health").textContent="API unavailable";$("health").className="status bad"}
}
load();