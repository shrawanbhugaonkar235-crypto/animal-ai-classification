const API_BASE=(window.ANIMAL_AI_API||"").replace(/\/$/,"");
const api=path=>API_BASE+path;
const $=id=>document.getElementById(id);
let selected=null;
let detections=[];

$("chooseBtn").addEventListener("click",()=>$("fileInput").click());
$("changeBtn").addEventListener("click",()=>$("fileInput").click());
$("fileInput").addEventListener("change",()=>{const f=$("fileInput").files[0];if(f)setFile(f)});
$("analyzeBtn").addEventListener("click",analyze);
$("newPhotoBtn").addEventListener("click",resetApp);

["search","category","confidence"].forEach(id=>$(id).addEventListener("input",renderDetections));

["dragenter","dragover"].forEach(evt=>$("dropZone").addEventListener(evt,e=>{e.preventDefault();$("dropZone").classList.add("drag")}));
["dragleave","drop"].forEach(evt=>$("dropZone").addEventListener(evt,e=>{e.preventDefault();$("dropZone").classList.remove("drag")}));
$("dropZone").addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f)setFile(f)});
$("dropZone").addEventListener("click",e=>{if(e.target===$("dropZone")||e.target.tagName==="H2"||e.target.tagName==="P"||e.target.classList.contains("upload-icon"))$("fileInput").click()});

function setFile(file){
  if(!["image/jpeg","image/png","image/webp"].includes(file.type))return showError("Please choose JPG, JPEG, PNG or WEBP.");
  selected=file;
  $("fileName").textContent=file.name+" · "+(file.size/1024/1024).toFixed(2)+" MB";
  $("preview").src=URL.createObjectURL(file);
  $("dropZone").classList.add("hidden");
  $("selectedPanel").classList.remove("hidden");
  $("results").classList.add("hidden");
  hideError();
}

function resetApp(){
  selected=null;
  $("fileInput").value="";
  $("preview").removeAttribute("src");
  $("selectedPanel").classList.add("hidden");
  $("dropZone").classList.remove("hidden");
  $("results").classList.add("hidden");
  detections=[];
  hideError();
}

function showError(message){$("error").textContent=message;$("error").classList.remove("hidden")}
function hideError(){$("error").textContent="";$("error").classList.add("hidden")}

async function analyze(){
  if(!selected)return showError("Choose a photo first.");
  $("loading").classList.remove("hidden");
  $("analyzeBtn").disabled=true;
  hideError();
  try{
    const fd=new FormData();
    fd.append("file",selected);
    const response=await fetch(api("/api/predict"),{method:"POST",body:fd});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||"Prediction failed.");
    detections=data.detections||[];
    $("total").textContent=data.total_detections;
    $("land").textContent=data.summary.land_animals;
    $("sea").textContent=data.summary.sea_animals;
    $("birds").textContent=data.summary.birds;
    $("avg").textContent=data.summary.average_confidence+"%";
    $("time").textContent=data.processing_time_ms+" ms";
    $("resultModel").textContent=data.model_name;
    $("resultImage").src=data.annotated_image;
    $("results").classList.remove("hidden");
    renderDetections();
    $("results").scrollIntoView({behavior:"smooth",block:"start"});
  }catch(error){showError(error.message||"Could not analyze the photo.")}finally{
    $("loading").classList.add("hidden");
    $("analyzeBtn").disabled=false;
  }
}

function renderDetections(){
  const q=$("search").value.trim().toLowerCase();
  const cat=$("category").value;
  const min=Number($("confidence").value||0);
  const rows=detections.filter(x=>x.name.toLowerCase().includes(q)&&(cat==="All"||x.category===cat)&&x.confidence_percent>=min);
  $("detectionList").innerHTML=rows.length?rows.map(x=>`<div class="detection-item"><div class="item-top"><div><b>${escapeHtml(x.name)}</b><small>${escapeHtml(x.category)} · ${escapeHtml(x.status)}</small></div><b>${x.confidence_percent}%</b></div><div class="progress"><span style="width:${Math.min(100,x.confidence_percent)}%"></span></div><div class="bbox">Box: [${x.bbox.x1}, ${x.bbox.y1}] → [${x.bbox.x2}, ${x.bbox.y2}]</div></div>`).join(""):'<div class="empty">No detections match the current filters.</div>';
}

function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]))}

async function loadStatus(){
  try{
    const health=await (await fetch(api("/api/health"))).json();
    $("apiStatus").textContent=health.model_loaded?"AI API ready":"API online · model unavailable";
    $("apiStatus").previousElementSibling.classList.toggle("ok",health.model_loaded);
    $("apiStatus").previousElementSibling.classList.toggle("bad",!health.model_loaded);
    $("modelChip").textContent=health.model_loaded?"Model: "+health.model_name:"Model: not loaded";
  }catch{
    $("apiStatus").textContent="AI API unavailable";
    $("apiStatus").previousElementSibling.classList.add("bad");
    $("modelChip").textContent="Model: unavailable";
  }
}
loadStatus();