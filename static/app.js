const get = id => document.getElementById(id);
const shell = get("workspace");
const modal = get("dialog");
const notice = get("status");
const state = {
    panels:[], samples:[], tools:[], panel:null, photos:[], reports:[], surveys:[], activity:[],
    chosen:null, selection:[], tab:"inspect", tool:"transmission", options:{}, region:[0,0,1,1],
    draft:null, saved:null, preview:null, pictured:null, draw:null, alignment:[], alignTrial:null,
    sourceView:null, survey:null, busy:false, loadingPanel:null,
    alignMode:"corners", localLines:[], localTrial:null, localParams:{tolerance_mm:1.5,smoothing:.35}
};
const api = () => "/api/panels/"+state.panel.id;
const capture = () => state.photos.find(photo => photo.id === state.chosen);
const definition = () => state.tools.find(tool => tool.key === state.tool);
const active = () => state.saved || state.draft;
const arr = item => Array.isArray(item) ? item : [];
const text = input => String(input??"").replace(/[&<>'"]/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[ch]));
function val(input) {
    if (input === undefined || input === null) return "—";
    if (typeof input === "number") return String(Number(input.toFixed(5)));
    if (typeof input === "boolean") return input ? "Yes" : "No";
    if (Array.isArray(input)) return input.every(x=>typeof x!=="object") ? input.map(val).join(" / ") : input.length+" items";
    return typeof input === "object" ? JSON.stringify(input) : input;
}
function msg(value,error=false) {
    notice.textContent=value;
    notice.dataset.error=String(error);
    notice.hidden=false;
    clearTimeout(msg.timer);
    msg.timer=setTimeout(()=>notice.hidden=true,error?11000:4300);
}
async function rpc(url,method="GET",body) {
    const config={method,headers:{}};
    if (body instanceof FormData) config.body=body;
    else if (body!==undefined) {
        config.headers["Content-Type"]="application/json";
        config.body=JSON.stringify(body);
    }
    const response=await fetch(url,config);
    const result=await response.json();
    if (!response.ok) throw new Error(result.error||"Request failed");
    return result;
}
function b(title,action,key="",classes="") {
    return '<button type="button" class="'+text(classes)+'" data-act="'+text(action)+'" data-key="'+text(key)+'">'+text(title)+'</button>';
}
function a(title,href,classes="tight") {
    return '<a class="button '+classes+'" href="'+text(href)+'" download>'+text(title)+'</a>';
}
function input(title,name,value="",type="text",step="") {
    return '<label>'+text(title)+'<input name="'+text(name)+'" type="'+text(type)+'" value="'+text(value)+'" '+step+'></label>';
}
function selection(title,name,entries,selected) {
    return '<label>'+text(title)+'<select name="'+text(name)+'">'+entries.map(([key,name])=>'<option value="'+text(key)+'" '+(selected===key?"selected":"")+">"+text(name)+"</option>").join("")+'</select></label>';
}
function notes(title,key,value="") {
    return '<label>'+text(title)+'<textarea name="'+text(key)+'" rows="3">'+text(value)+'</textarea></label>';
}
function dialog(title,content) {
    modal.replaceChildren();
    modal.innerHTML='<div class="dialog-head"><h2>'+text(title)+'</h2>'+b("Close","close","","tight")+'</div>'+content;
    if(!modal.open) modal.showModal();
}
function form(kind,body,save) {
    return '<form data-form="'+kind+'">'+body+'<button class="warm" type="submit">'+text(save)+'</button></form>';
}
function close() { if (modal.open) modal.close(); }
function dataSource(asset) { return asset?.url || (asset?.data ? "data:"+asset.mime+";base64,"+asset.data : ""); }
function detail(value,title="Readings and evidence") {
    return '<details class="disclosure"><summary>'+text(title)+'</summary><pre>'+text(JSON.stringify(value,null,2))+'</pre></details>';
}
function metrics(rows) {
    return '<div class="metric-grid">'+Object.entries(rows).slice(0,14).filter(([,v])=>v===null||typeof v!=="object"||Array.isArray(v)).map(([k,v])=>'<div><span>'+text(k.replaceAll("_"," "))+'</span><strong>'+text(val(v))+'</strong></div>').join("")+"</div>";
}
async function library() {
    const [bank,examples]=await Promise.all([rpc("/api/library"),rpc("/api/samples")]);
    state.panels=bank.panels; state.samples=examples.samples; state.panel=null;
    state.loadingPanel=null; state.busy=false; render();
}
function resetFrame() { state.draft=null; state.saved=null; state.pictured=null; state.preview=null; state.draw=null; state.sourceView=null; }
async function openPanel(id,keep=false) {
    state.loadingPanel=id;
    const result=await rpc("/api/panels/"+id);
    if (state.loadingPanel!==id) return;
    if(!keep || state.panel?.id!==id) {
        resetFrame();
        state.selected=null; state.region=[0,0,1,1]; state.tab="inspect"; state.survey=null;
        state.chosen=result.photos[0]?.id;
        state.selection=state.chosen?[state.chosen]:[];
        state.options={};
    }
    Object.assign(state,result);
    await currentMatrix();
    render();
}
function libraryPage() {
    return '<section class="library"><div class="library-head"><h1>Panel collection</h1><p class="quiet">Backlit glazing inspection and evidence</p></div><div class="panels"><h2>Saved panels</h2>'+
        (state.panels.length?state.panels.map(panel=>'<button class="panel-row" data-act="panel" data-key="'+panel.id+'"><strong>'+text(panel.title)+'</strong><span>'+text(panel.accession)+' · '+panel.photos+' captures · '+panel.reports+' inspections</span></button>').join(""):'<p class="quiet">Open a specimen or create a panel to inspect your own captures.</p>')+
        '</div><h2>Demonstration specimens</h2><div class="sample-block">'+state.samples.map(item=>'<article class="sample-card"><img src="'+item.thumbnail+'" alt=""><div class="sample-copy"><h3>'+text(item.title)+'</h3><p class="quiet">'+text(item.caption)+'</p><div class="row">'+b("Load specimen","sample",item.key,"warm")+'<span class="role">Synthetic</span></div></div></article>').join("")+'</div></section>';
}
function photosRail() {
    const panel=state.panel;
    return '<aside class="rail" aria-label="Captures"><div><h1>'+text(panel.title)+'</h1><p class="quiet">'+text(panel.accession)+" · "+panel.width_mm+' × '+panel.height_mm+' mm</p>'+(panel.synthetic?'<span class="role">Synthetic</span>':'')+'</div>'+
        '<div class="row">'+b("Add capture","add","","warm")+b("Panel details","edit-panel","","tight")+'</div>'+
        '<div><h2>Camera captures</h2><p class="quiet">Use the ticks to select inputs.</p></div><div>'+state.photos.map(photo=>'<div class="photo-row"><label><input type="checkbox" data-select-photo="'+photo.id+'" aria-label="Use '+text(photo.data.title)+'" '+(state.selection.includes(photo.id)?"checked":"")+'></label><button class="select-photo" data-act="capture" data-key="'+photo.id+'" aria-pressed="'+(state.chosen===photo.id)+'"><img src="/api/photos/'+photo.id+'/view/preview" alt=""><span>'+text(photo.data.title)+'</span></button></div>').join("")+'</div>'+
        (capture()?'<div class="row">'+a("Original","/api/photos/"+state.chosen+"/view/original")+b("Capture details","edit-capture","","tight")+b("Align corners","alignment","","tight")+'</div>':"")+'</aside>';
}
function overlay() {
    let points=state.draw?.cameraPoints||state.draw?.points||[];
    if(state.tab==="align"&&state.alignMode==="corners"&&!state.draw) points=state.alignment;
    let marks="";
    if(points.length) marks='<polyline points="'+points.map(([x,y])=>x*1000+","+y*1000).join(" ")+'" fill="none" stroke="#f3d376" stroke-width="2"/>'+points.map(([x,y])=>'<circle cx="'+x*1000+'" cy="'+y*1000+'" r="6" fill="#f3d376" stroke="#323b3d" stroke-width="1"/>').join("");
    if(state.draw?.shape==="box" && points.length===2) {
        const [p,q]=points;
        marks+='<rect x="'+Math.min(p[0],q[0])*1000+'" y="'+Math.min(p[1],q[1])*1000+'" width="'+Math.abs(p[0]-q[0])*1000+'" height="'+Math.abs(p[1]-q[1])*1000+'" fill="none" stroke="#f3d376" stroke-width="2"/>';
    }
    if(state.tab==="align"&&state.alignMode==="local"&&!state.draw) {
        const statusOf=i=>state.localTrial?.lines?.[i]?.status;
        const toCamera=pt=>{
            const m=state.activeMatrix;
            if(!m) return pt;
            const [x,y]=pt,[a,b,c]=m;
            const den=c[0]*x+c[1]*y+c[2];
            return [(a[0]*x+a[1]*y+a[2])/den,(b[0]*x+b[1]*y+b[2])/den];
        };
        marks+=state.localLines.map((line,i)=>{
            const status=statusOf(i);
            const stroke=status==="within"||status==="frame-ok"?"#7fd8ad":status==="unseen"?"#f1c863":status?"#f4816e":"#7fb6e8";
            const pts=[toCamera(line.from),toCamera(line.to)].map(p=>p[0]*1000+","+p[1]*1000).join(" ");
            return '<polyline points="'+pts+'" fill="none" stroke="'+stroke+'" stroke-width="2.5"/>'+
                [line.from,line.to].map(p=>{const q=toCamera(p);return '<circle cx="'+q[0]*1000+'" cy="'+q[1]*1000+'" r="6" fill="'+stroke+'" stroke="#323b3d" stroke-width="1"/>';}).join("");
        }).join("");
    }
    return '<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">'+marks+'</svg>';
}
function stage() {
    const result=active();
    const item=result?.assets.find(part=>part.key===state.pictured&&part.mime==="image/png");
    let natural=state.chosen?"/api/photos/"+state.chosen+"/view/preview":"";
    let picture=state.sourceView||state.preview||dataSource(item)||natural;
    if(state.draw||state.tab==="align") picture=(state.alignTrial||state.localTrial)&&state.preview?state.preview:natural;
    if(state.tab==="align"&&(state.alignTrial||state.localTrial)) picture=state.preview;
    const labelText=picture===natural?"Camera preview":state.tab==="align"?"Aligned panel preview":item?.key?.replaceAll("-"," ")||"Registered panel preview";
    return '<section class="image-area" aria-label="Panel image"><div class="image-bar"><h2>'+text(labelText)+'</h2><div class="row">'+b("Camera","show-camera","","tight")+b("Panel plane","panel-plane","","tight")+'</div></div>'+
        (state.draw?'<p class="notice">'+text(state.draw.prompt)+" "+state.draw.points.length+" point(s). "+b("Use marks","finish-draw","","tight")+b("Cancel","cancel-draw","","tight")+'</p>':"")+
        '<div class="stage">'+(picture?'<div class="plate '+(state.draw||state.tab==="align"?"drawing":"")+'" id="plate"><img src="'+picture+'" alt="'+text(labelText)+'">'+overlay()+'</div>':'<div class="empty">Add a capture to open the image workbench.</div>')+"</div>"+
        (result?assets(result):"")+"</section>";
}
function assets(result) {
    return '<div class="asset-tabs">'+result.assets.filter(asset=>asset.mime==="image/png").map(part=>'<button type="button" data-act="asset" data-key="'+text(part.key)+'" aria-pressed="'+(state.pictured===part.key)+'">'+text(part.key.replaceAll("-"," "))+'</button>').join("")+
        result.assets.filter(asset=>asset.mime!=="image/png").map(part=>a(part.filename,dataSource(part))).join("")+'</div>';
}
function optionValue(field) { return state.options[field.key]??field.default; }
function setting(field) {
    const current=optionValue(field);
    if (field.input==="select") return selection(field.label,field.key,field.options.map(v=>[v,v]),current);
    if (field.input==="position") return '<div class="pick">'+input(field.label,field.key,JSON.stringify(current))+b("Mark","draw",field.key,"tight")+'</div>';
    return input(field.label,field.key,current,"number","step=\""+field.options+"\"");
}
function resultBox() {
    const result=active();
    if(!result) return "";
    const kind=state.saved?"Retained inspection":"Preview";
    const extra=state.saved?'<div class="row">'+a("Evidence ZIP","/api/inspections/"+state.saved.id+"/packet")+a("Results-only ZIP","/api/inspections/"+state.saved.id+"/packet?originals=0")+a("Print sheet","/api/inspections/"+state.saved.id+"/sheet")+b("Review status","review-dialog","","tight")+b("Compare source settings","source-audit","","tight")+'</div>':
        '<div class="row">'+b("Retain inspection","retain-dialog","","warm")+'</div>';
    return '<div class="output"><h3>'+kind+'</h3>'+metrics(result.summary||result.result)+extra+detail(result.result)+detail(result.facts,"Source lineage")+'</div>';
}
function inspection() {
    const current=definition();
    if(!current) return "";
    const groups=Object.groupBy ? Object.groupBy(state.tools,tool=>tool.area) : state.tools.reduce((g,tool)=>{(g[tool.area]??=[]).push(tool);return g;},{});
    const options=Object.entries(groups).map(([name,tools])=>'<optgroup label="'+text(name)+'">'+tools.map(tool=>'<option value="'+tool.key+'" '+(tool.key===state.tool?"selected":"")+'>'+text(tool.title)+'</option>').join("")+'</optgroup>').join("");
    const drawRegion='<div class="pick">'+input("Panel window","region",JSON.stringify(state.region))+b("Mark","draw","region","tight")+'</div>';
    const surveyPick=state.tool==="survey"?selection("Survey","survey",state.surveys.map(x=>[x.id,x.name]),state.survey?.id):"";
    return '<div class="settings"><label>Tool<select id="tool">'+options+'</select></label><p class="quiet">'+text(current.explanation)+'</p>'+
        '<form data-form="analysis">'+drawRegion+current.fields.map(setting).join("")+surveyPick+
        '<button type="submit" class="warm" '+(state.busy||!state.selection.length?"disabled":"")+'>Run inspection</button></form></div>'+resultBox();
}
function alignPane() {
    const current=capture();
    if(!current) return '<p>Select a camera capture.</p>';
    const revisions=state.registration?.revisions||[];
    const params=state.alignParams||{title:"Panel outline",model:"projective"};
    const corner='<h2>Outline alignment</h2><p class="quiet">Click all four camera corners clockwise from the upper left. The outer frame and its measured edge lengths stay on this registration.</p>'+
        '<p>'+state.alignment.length+' / 4 corners</p><div class="row">'+b("Clear corners","clear-align","","tight")+'</div>'+
        '<form data-form="align">'+input("Alignment name","title",params.title)+
        selection("Projection","model",[["projective","Perspective"],["affine","Affine"]],params.model)+
        '<div class="row"><button class="warm" '+(state.alignment.length!==4?"disabled":"")+' type="submit">Preview alignment</button>'+
        (state.alignTrial?b("Use alignment","retain-align","","warm"):"")+'</div></form>'+
        (state.alignTrial?metrics({coverage:state.alignTrial.coverage,error:state.alignTrial.alignment.error_rms}):"");
    const local=localAlignPane();
    return '<h2>Registration</h2><div class="tabs" role="tablist">'+
        '<button role="tab" aria-selected="'+(state.alignMode==="corners")+'" data-act="align-mode" data-key="corners">Four corners</button>'+
        '<button role="tab" aria-selected="'+(state.alignMode==="local")+'" data-act="align-mode" data-key="local">Local measured lines</button></div>'+
        (state.alignMode==="corners"?corner:local)+
        '<h3>Saved alignments</h3><div class="align-list">'+b("Whole image","activate-align","")+
        revisions.map(item=>b(item.data.title+(item.data.model==="local"?" · local":"")+(item.id===state.registration.active?" · Active":""),"activate-align",item.id)).join("")+"</div>";
}
const LOCAL_STATUS_HINT={
    "within":"Reading matched after the local correction",
    "frame-ok":"Frame edge anchor agrees with the four-corner registration",
    "disagreement":"The ruler reading still differs beyond tolerance after the fit",
    "frame-conflict":"A frame-edge reading conflicts with the kept four-corner frame and was excluded",
    "unseen":"An endpoint falls outside captured support; the reading cannot be checked"
};
function localAlignPane() {
    const lines=state.localLines;
    const trial=state.localTrial;
    const statusRows=trial?trial.lines.map((row,i)=>{
        const color={within:"#7fd8ad","frame-ok":"#60be9e",disagreement:"#f4816e","frame-conflict":"#e05448",unseen:"#f1c863"}[row.status];
        return '<div class="scoped"><strong style="color:'+color+'">'+text(row.name)+' · '+text(row.status)+'</strong>'+
            '<p class="quiet">'+text(LOCAL_STATUS_HINT[row.status])+'</p>'+
            '<p class="quiet">ruler '+text(row.measured_mm)+' mm · four-corner '+text(row.baseline_mm)+' mm · corrected '+
            text(row.corrected_mm)+' mm (residual '+text(row.corrected_difference_mm)+' mm)</p></div>';
    }).join(""):"";
    const editor=lines.length?'<div class="mark-grid">'+lines.map((line,i)=>
        '<div class="scoped"><strong>'+text(line.name)+(line.on_frame?' · frame anchor':' · interior')+'</strong>'+
        '<p class="quiet">'+JSON.stringify(line.from)+' → '+JSON.stringify(line.to)+'</p>'+
        '<div class="row"><label class="fieldgrid">measured mm<input type="number" step="0.01" data-line-mm="'+i+'" value="'+text(line.measured_mm??"")+'"></label>'+
        b("Remove","remove-local-line",i,"tight")+'</div></div>').join("")+'</div>':
        '<p class="quiet">Draw a measured line across the squeezed middle panes, then type the millimetres from your ruler. Lines on the outer frame only anchor the border and never move it.</p>';
    return '<p class="quiet">Correct a locally squeezed area while keeping the four-corner frame fixed. Fill every measured value; blank or conflicting readings are flagged separately.</p>'+
        '<div class="row">'+b("Draw interior line","add-local-line","","warm")+b("Draw frame anchor","add-frame-line","","tight")+'</div>'+
        editor+
        '<form data-form="local-fit">'+input("Fit name","title","Local measured lines")+
        input("Length tolerance (mm)","tolerance_mm",state.localParams.tolerance_mm,"number","step=.05")+
        input("Smoothing strength (0–1)","smoothing",state.localParams.smoothing,"number","step=.05")+
        '<div class="row"><button class="warm" type="submit" '+(lines.length?"":"disabled")+'>Preview local fit</button>'+
        (trial?b("Use this local fit","retain-local","","warm"):"")+'</div></form>'+
        (trial?metrics(trial.summary):"")+(statusRows?'<div class="scroll">'+statusRows+'</div>':"");
}
function recordPane() {
    return '<h2>Retained inspections</h2>'+(state.reports.length?a("Inspection index PDF",api()+"/inspection-index"):"")+'<div class="reports">'+(state.reports.length?state.reports.map(item=>'<button class="report-row" data-act="record" data-key="'+item.id+'"><strong>'+text(item.name)+'</strong><span>'+text(item.review.state)+'</span></button>').join(""):'<p class="quiet">Retain a preview to keep its readings.</p>')+'</div>'+resultBox()+detail(state.activity,"Activity");
}
function surveyPane() {
    const record=state.survey;
    const entries=record?.geometry?.marks||[];
    return '<h2>Survey marks</h2><div class="row">'+b("New survey","new-survey","","warm")+'</div><div class="reports">'+
        state.surveys.map(row=>b(row.name,"open-survey",row.id,"tight")).join("")+"</div>"+
        (record?'<div class="notice"><strong>'+text(record.name)+'</strong><div class="row">'+b("Add mark","add-mark","","tight")+b("Save survey","save-survey","","warm")+b("Measure on capture","survey-measure","","tight")+'</div></div><div class="mark-grid">'+entries.map(mark=>'<div class="row"><span>'+text(mark.title)+' · '+text(mark.kind)+'</span>'+b("Remove","remove-mark",mark.key,"tight")+'</div>').join("")+'</div>':"")+resultBox();
}
function toolRail() {
    const sections=[["inspect","Inspect"],["align","Alignment"],["survey","Survey"],["record","Records"]];
    const selected={inspect:inspection,align:alignPane,record:recordPane,survey:surveyPane};
    return '<aside class="tools" aria-label="Controls"><div class="tabs" role="tablist">'+sections.map(([key,title])=>'<button role="tab" aria-selected="'+(state.tab===key)+'" data-act="tab" data-key="'+key+'">'+title+'</button>').join("")+'</div>'+selected[state.tab]()+'</aside>';
}
function render() { shell.innerHTML=state.panel?'<div class="bench">'+photosRail()+stage()+toolRail()+'</div>':libraryPage(); }
function keepPresets() {
    const f=shell.querySelector('form[data-form="analysis"]');
    if(!f) return;
    const formData=new FormData(f);
    for(const field of definition().fields) {
        const raw=formData.get(field.key);
        if(raw===null) continue;
        try { state.options[field.key]=field.input==="position"?JSON.parse(raw):field.input==="number"?Number(raw):raw; }
        catch { throw Error("Enter coordinates as a JSON list or use Mark."); }
    }
    try {state.region=JSON.parse(formData.get("region"));} catch {throw Error("Enter four region bounds or use Mark.");}
    if(formData.get("survey")) state.survey=state.surveys.find(item=>item.id===formData.get("survey"));
}
function requestOrder() {
    const tool=definition();
    const options={...state.options};
    if(tool.key==="survey") {
        if(!state.survey?.geometry.marks?.length) throw Error("Choose a survey with at least one mark.");
        options.marks=state.survey.geometry.marks;
    }
    const captures=state.tab==="survey"&&state.tool==="survey"?[state.chosen]:state.selection;
    return {captures,region:state.region,options, resolution:700};
}
async function run() {
    keepPresets();
    const payload=requestOrder();
    const token=state.panel.id+"-"+state.tool;
    state.busy=true; render();
    try {
        const work=await rpc(api()+"/tools/"+state.tool,"POST",payload);
        if(token!==state.panel?.id+"-"+state.tool) return;
        state.draft={...work, order:payload};
        state.saved=null; state.preview=null; state.sourceView=null;
        state.pictured=state.draft.assets.find(part=>part.mime==="image/png")?.key;
    } finally {state.busy=false; render();}
}
async function alignOpen() {
    state.tab="align"; state.alignTrial=null; state.preview=null;
    state.alignment=[]; state.alignParams={title:"Panel outline",model:"projective"};
    if(state.alignMode===undefined) state.alignMode="corners";
    if(!Array.isArray(state.localLines)) state.localLines=[];
    state.localTrial=null;
    state.registration=capture()?await rpc("/api/photos/"+state.chosen+"/alignments"):null;
    render();
}
function dialogPanel(edit=false) {
    const old=edit?state.panel:{};
    dialog(edit?"Panel details":"New panel",form(edit?"edit-panel":"new-panel",
        input("Panel name","title",old.title||"","text","required")+input("Accession","accession",old.accession||"")+
        '<div class="fieldgrid">'+input("Width (mm)","width_mm",old.width_mm||160,"number","step=.1 required")+
        input("Height (mm)","height_mm",old.height_mm||120,"number","step=.1 required")+'</div>'+notes("Notes","note",old.note||""),edit?"Save panel":"Create panel"));
}
function captureFields(meta={}) {
    return input("Capture title","title",meta.title||"")+selection("Capture kind","role",["transmitted","comparison","reflected","flat"].map(v=>[v,v]),meta.role||"transmitted")+
        '<div class="fieldgrid">'+input("Exposure EV","exposure_ev",meta.exposure_ev??0,"number","step=.05")+
        input("Black code","black_code",meta.black_code??0,"number","step=.001")+'</div>'+
        input("Lighting label","illuminant",meta.illuminant||"")+notes("Notes","note",meta.note||"");
}
function startLocalLine(onFrame) {
    if(!capture()?.active_alignment) {
        msg("Save a four-corner alignment first so drawn marks map to the panel plane.",true);
        return;
    }
    drawing(onFrame?"frame-line":"local-line");
}
function drawing(key,inputType=null) {
    if(state.tab==="inspect") keepPresets();
    const localKeys=new Set(["local-line","frame-line"]);
    const current=localKeys.has(key)?"line":(inputType || (key==="path"?"line":key==="boundary"?"polygon":["region","reference","background","color_reference","middle"].includes(key)?"box":"point"));
    const rules={point:"Click a location on the camera image.",line:"Click the two ends of the measured line.",box:"Click two opposite corners.",polygon:"Click successive polygon vertices."};
    state.draw={key,shape:current,prompt:rules[current]||rules.line,points:[],cameraPoints:[]}; state.sourceView=null;
    render();
}
function finishDraw() {
    const selected=state.draw;
    if(!selected) return;
    const points=selected.points,limit={point:1,line:2,box:2,polygon:3}[selected.shape];
    if(points.length<limit) throw Error("Mark more points before using this selection.");
    const place=selected.shape==="point"?points[0]:selected.shape==="box"?
        [Math.min(points[0][0],points[1][0]),Math.min(points[0][1],points[1][1]),Math.max(points[0][0],points[1][0]),Math.max(points[0][1],points[1][1])]:points;
    if(selected.key==="region") state.region=place;
    else if(selected.key==="new-survey-mark") {
        const mark={...selected.mark,points:selected.shape==="point"?[place]:place};
        state.survey.geometry.marks.push(mark);
    } else if(selected.key==="local-line"||selected.key==="frame-line") {
        const onFrame=selected.key==="frame-line";
        state.localLines.push({name:(onFrame?"Frame anchor ":"Measured line ")+(state.localLines.length+1),
            from:place[0],to:place[1],measured_mm:null,on_frame:onFrame});
        state.localTrial=null;
    } else state.options[selected.key]=place;
    state.draw=null;
    render();
}
function pointEvent(event) {
    const zone=event.target.closest("#plate");
    if(!zone||(!state.draw&&state.tab!=="align")) return false;
    const box=zone.querySelector("img").getBoundingClientRect();
    if(!box.width||!box.height) return true;
    let point=[(event.clientX-box.left)/box.width,(event.clientY-box.top)/box.height].map(v=>Number(Math.max(0,Math.min(1,v)).toFixed(6)));
    const cameraPoint=point;
    const localDrawing=state.draw&&(state.draw.key==="local-line"||state.draw.key==="frame-line");
    if (state.draw && (state.tab !== "align" || localDrawing) && capture()?.active_alignment) {
        const m=state.activeMatrix;
        if (!m) throw Error("Load the panel plane first or reactivate its alignment.");
        const [a,b,c]=m, [x,y]=point;
        const aa=a[0]-x*c[0], ab=a[1]-x*c[1], ad=x*c[2]-a[2];
        const bb=b[0]-y*c[0], bc=b[1]-y*c[1], bd=y*c[2]-b[2];
        const det=aa*bc-ab*bb;
        if (Math.abs(det)<1e-8) throw Error("Point cannot be mapped to the panel.");
        point=[(ad*bc-ab*bd)/det,(aa*bd-ad*bb)/det].map(v=>Number(v.toFixed(6)));
        if (point.some(v=>v<0||v>1)) throw Error("That point is outside the aligned panel.");
    }
    if(state.draw) {
        const twoEnd=state.draw.key==="local-line"||state.draw.key==="frame-line";
        const limit=twoEnd?2:{point:1,box:2,line:90,polygon:120}[state.draw.shape];
        if(state.draw.points.length>=limit) {state.draw.points=[];state.draw.cameraPoints=[];}
        state.draw.points.push(point);
        state.draw.cameraPoints.push(cameraPoint);
        if(twoEnd&&state.draw.points.length===2) return finishDraw()??true;
        if(state.draw.shape==="point"||state.draw.shape==="box"&&state.draw.points.length===2) return finishDraw()??true;
    } else {
        if(state.alignment.length>=4) return true;
        state.alignment.push(point); state.alignTrial=null;
    }
    render(); return true;
}
async function currentMatrix() {
    state.activeMatrix=null;
    if(!capture()?.active_alignment) return;
    const reg=await rpc("/api/photos/"+state.chosen+"/alignments");
    const item=reg.revisions.find(x=>x.id===reg.active);
    if(item) state.activeMatrix=item.data.matrix;
}
async function withCapture(id) {
    state.chosen=id;
    if(state.selection.length<=1) state.selection=[id];
    resetFrame();
    state.localLines=[]; state.localTrial=null; state.localDraft=null;
    await currentMatrix();
    if(state.tab==="align") await alignOpen();
    else render();
}
async function showRecord(id) {
    const {record}=await rpc("/api/inspections/"+id);
    state.saved=record; state.draft=null; state.preview=null; state.sourceView=null;
    state.pictured=record.assets.find(asset=>asset.mime==="image/png")?.key;
    state.tab="record"; render();
}
async function previewPlane() {
    if(!capture()) return;
    const row=capture();
    let align={corners:[[0,0],[1,0],[1,1],[0,1]],title:"Whole image",model:"projective"};
    if(row.active_alignment) {
        const remote=await rpc("/api/photos/"+row.id+"/alignments");
        const record=remote.revisions.find(item=>item.id===remote.active)?.data;
        if(record) align={title:record.title,model:record.model,corners:record.corners};
    }
    const result=await rpc("/api/photos/"+row.id+"/alignments","POST",{revision:row.revision,alignment:align,preview:true});
    state.sourceView=dataSource(result.image); render();
}
function activeCorners() {
    const row=capture();
    const found=state.registration?.revisions.find(item=>item.id===row?.active_alignment)?.data;
    return found?.corners||[[0,0],[1,0],[1,1],[0,1]];
}
function localSegments() {
    return state.localLines.map(line=>({name:line.name,from:line.from,to:line.to,
        measured_mm:line.measured_mm===null||line.measured_mm===""?null:Number(line.measured_mm)}));
}
async function previewLocalFit(formValues) {
    state.localParams={tolerance_mm:formValues.tolerance_mm,smoothing:formValues.smoothing};
    const row=capture();
    if(!row?.active_alignment) throw Error("Save a four-corner alignment before fitting measured lines.");
    const blank=state.localLines.filter(line=>line.measured_mm===null||line.measured_mm==="").map(line=>line.name);
    if(blank.length) throw Error("Fill the measured millimetres for: "+blank.join(", "));
    const alignment={title:formValues.title||"Local measured lines",model:"local",corners:activeCorners(),
        local:{segments:localSegments(),tolerance_mm:formValues.tolerance_mm,smoothing:formValues.smoothing,
               panel_mm:[state.panel.width_mm,state.panel.height_mm]}};
    const answer=await rpc("/api/photos/"+row.id+"/alignments","POST",
        {revision:row.revision,alignment,preview:true});
    state.localTrial=answer;
    state.localDraft=alignment;
    state.preview=dataSource(answer.image);
    render();
}
async function retainLocal() {
    if(!state.localDraft) return;
    const row=capture();
    await rpc("/api/photos/"+row.id+"/alignments",
        {revision:row.revision,alignment:state.localDraft,preview:false});
    msg("Local measured-line fit retained");
    await openPanel(state.panel.id,true);
    await currentMatrix();
    await alignOpen();
    state.alignMode="local";
    render();
}
const actions={
    panel:id=>openPanel(id),
    sample:async key=>{msg("Preparing specimen…");const data=await rpc("/api/samples/"+key,"POST");await openPanel(data.panel.id);},
    "close":close,
    "edit-panel":()=>dialogPanel(true),
    "add":()=>dialog("Add camera capture",form("upload",input("Image file","photo","","file","accept=\"image/*,.tif,.tiff\" required")+captureFields(),"Import capture")),
    "edit-capture":()=>dialog("Capture details",form("edit-capture",captureFields(capture().data),"Save capture")+detail(capture().image,"Camera metadata")),
    capture:withCapture,alignment:alignOpen,
    "align-mode":key=>{state.alignMode=key;state.draw=null;render();},
    "add-local-line":()=>startLocalLine(false),
    "add-frame-line":()=>startLocalLine(true),
    "remove-local-line":i=>{state.localLines.splice(Number(i),1);state.localTrial=null;render();},
    "retain-local":retainLocal,
    "show-camera":()=>{state.pictured=null;state.preview=null;state.sourceView=null;render();},
    "panel-plane":previewPlane,
    tab:async key=>{ if(key==="align") return alignOpen(); state.tab=key;state.draw=null; state.preview=null;render(); },
    asset:key=>{state.pictured=key;state.preview=null;state.sourceView=null;render();},
    draw:key=>drawing(key),
    "finish-draw":finishDraw,
    "cancel-draw":()=>{state.draw=null;render();},
    "clear-align":()=>{state.alignment=[];state.alignTrial=null;state.preview=null;render();},
    "retain-align":async()=>{ await rpc("/api/photos/"+state.chosen+"/alignments","POST",{alignment:state.alignParams,revision:capture().revision,preview:false});await openPanel(state.panel.id,true);await currentMatrix();await alignOpen();msg("Alignment retained"); },
    "activate-align":async key=>{await rpc("/api/photos/"+state.chosen+"/active-alignment","PUT",{revision:capture().revision,alignment_id:key||null});await openPanel(state.panel.id,true);await currentMatrix();await alignOpen();msg("Active alignment updated");},
    "retain-dialog":()=>dialog("Retain inspection",form("retain",input("Record name","name",state.draft.title),"Keep observed result")),
    record:showRecord,
    "source-audit":async()=>{const report=await rpc("/api/inspections/"+state.saved.id+"/source-audit");
        dialog("Retained and current source settings",'<p class="quiet">'+text(report.notes)+'</p>'+detail(report)+
          '<div class="row">'+report.sources.filter(source=>!source.missing).map(source=>a("Original · "+source.title,source.original_url)).join("")+'</div>');},

    "review-dialog":()=>dialog("Review retained inspection",form("review",input("Reviewer","name",state.saved.review.by||"","text","required")+
        selection("Decision","state",[["observed","Observed"],["confirmed","Confirmed"],["return","Needs revisit"]],state.saved.review.state)+notes("Review note","note",state.saved.review.note),"Save review")),
    "new-survey":()=>dialog("New survey",form("new-survey",input("Survey title","name","","text","required"),"Start marking")),
    "open-survey":key=>{state.survey=structuredClone(state.surveys.find(row=>row.id===key));render();},
    "add-mark":()=>dialog("Survey mark",form("add-mark",input("Mark label","title","","text","required")+
        selection("Shape","kind",[["point","Point"],["line","Line"],["polygon","Polygon"]],"polygon")+
        selection("Condition","role",["feature","repair","missing","datum","note"].map(v=>[v,v]),"feature"),"Draw mark")),
    "remove-mark":key=>{state.survey.geometry.marks=state.survey.geometry.marks.filter(mark=>mark.key!==key);render();},
    "save-survey":async()=>{const d=state.survey;const result=await rpc(api()+"/surveys","POST",{name:d.name,marks:d.geometry.marks,id:d.id,revision:d.revision});state.survey=structuredClone(result.survey);await openPanel(state.panel.id,true);msg("Survey retained");},
    "survey-measure":()=>{state.tool="survey";state.options={};return run();}
};
function fields(info, numeric) {
    const body={};
    for(const [key,v] of info.entries()) body[key]=numeric.includes(key)?Number(v):v;
    return body;
}
const submits={
    "new-panel":async f=>{const data=await rpc("/api/panels","POST",fields(f,["width_mm","height_mm"]));close();await openPanel(data.panel.id);},
    "edit-panel":async f=>{await rpc(api(),"PUT",{revision:state.panel.revision,panel:fields(f,["width_mm","height_mm"])});close();await openPanel(state.panel.id,true);},
    upload:async f=>{ const values=fields(f,["exposure_ev","black_code"]);delete values.photo;if(!values.title)delete values.title; const input=new FormData();input.set("photo",f.get("photo"));input.set("capture",JSON.stringify(values)); const {photo}=await rpc(api()+"/photos","POST",input);close();await openPanel(state.panel.id,true);await withCapture(photo.id);msg("Capture imported");},
    "edit-capture":async f=>{await rpc("/api/photos/"+state.chosen,"PUT",{revision:capture().revision,capture:fields(f,["exposure_ev","black_code"])});close();await openPanel(state.panel.id,true);msg("Capture settings saved");},
    analysis:run,
    align:async f=>{ const doc={title:f.get("title"),model:f.get("model"),corners:state.alignment}; const answer=await rpc("/api/photos/"+state.chosen+"/alignments","POST",{alignment:doc,revision:capture().revision,preview:true});state.alignParams=doc;state.alignTrial=answer;state.preview=dataSource(answer.image);render(); },
    "local-fit":async f=>{ await previewLocalFit({title:f.get("title"),tolerance_mm:Number(f.get("tolerance_mm")),smoothing:Number(f.get("smoothing"))}); },
    retain:async f=>{const candidate=state.draft;const data=await rpc(api()+"/tools/"+candidate.kind,"POST",{...candidate.order,save:true,preview:candidate.fingerprint,name:f.get("name")});close();await openPanel(state.panel.id,true);await showRecord(data.record.id);msg("Inspection retained");},
    review:async f=>{const id=state.saved.id;await rpc("/api/inspections/"+id+"/review","PUT",{revision:state.saved.revision,...fields(f,[])});close();await openPanel(state.panel.id,true);await showRecord(id);},
    "new-survey":f=>{state.survey={name:f.get("name"),geometry:{marks:[]}};close();render();},
    "add-mark":f=>{const doc=fields(f,[]);close();drawing("new-survey-mark",doc.kind);state.draw.mark={...doc,key:crypto.randomUUID().replaceAll("-","").slice(0,16)};},
    verify:async f=>{const result=await rpc("/api/packets/verify","POST",f);dialog("Packet verified",metrics({files:result.files,captures:result.cited_captures,originals:result.source_photographs})+detail(result));},
    target:async f=>{const answer=await fetch("/api/scale-target",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(fields(f,["width_mm","height_mm","ticks_mm"]))});if(!answer.ok) throw Error((await answer.json()).error);const url=URL.createObjectURL(await answer.blob());const link=document.createElement("a");link.href=url;link.download="vitrail-scale-target.svg";link.click();setTimeout(()=>URL.revokeObjectURL(url),1500);close();}
};
function task(work) { return Promise.resolve().then(work).catch(error=>msg(error.message,true)); }
document.body.addEventListener("click",e=>task(async()=>{
    if(pointEvent(e)) return;
    const pressed=e.target.closest("[data-act]");
    if(pressed && actions[pressed.dataset.act]) await actions[pressed.dataset.act](pressed.dataset.key);
}));
document.body.addEventListener("submit",e=>{
    const form=e.target.closest("[data-form]");
    if(!form) return;
    e.preventDefault();
    const data=new FormData(form);
    task(()=>submits[form.dataset.form](data));
});
shell.addEventListener("change",e=>task(()=>{
    const id=e.target.dataset.selectPhoto;
    if(id) {
        const checked=e.target.checked;
        if(checked&&state.selection.length>=5) {
            e.target.checked=false;throw Error("Select up to five captures.");
        }
        state.selection=checked?[...state.selection,id]:state.selection.filter(key=>key!==id);
        const go=shell.querySelector('form[data-form="analysis"] button[type=submit]');
        if(go) go.disabled=!state.selection.length;
    }
    if(e.target.id==="tool") {
        state.tool=e.target.value;state.options={};resetFrame();render();
    }
    const lineIndex=e.target.dataset.lineMm;
    if(lineIndex!==undefined) {
        const raw=e.target.value;
        state.localLines[Number(lineIndex)].measured_mm=raw===""?null:Number(raw);
        state.localTrial=null;
    }
}));
get("home").onclick=()=>task(library);
get("new").onclick=()=>dialogPanel(false);
get("verify").onclick=()=>dialog("Verify an evidence packet",form("verify",input("Evidence ZIP","archive","","file","accept=\".zip\" required"),"Verify packet"));
get("scale").onclick=()=>dialog("Print calibration target",form("target",input("Target label","name","Panel reference")+
    '<div class="fieldgrid">'+input("Width (mm)","width_mm",120,"number","step=.1")+input("Height (mm)","height_mm",80,"number","step=.1")+'</div>'+
    input("Tick interval (mm)","ticks_mm",5,"number","step=.1"),"Download 1:1 SVG"));
task(async()=>{state.tools=(await rpc("/api/tools")).tools;await library();});
