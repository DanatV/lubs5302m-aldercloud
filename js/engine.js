/* All calculations are local. See METHODS.md for definitions and approximations. */
function createEngine() {
  'use strict';
  const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const sigmoid=x=>1/(1+Math.exp(-clamp(x,-35,35)));
  function rng(seed=5302){return ()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
  function shuffle(a,r){a=a.slice();for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function quantile(a,q){let s=a.slice().sort((x,y)=>x-y),p=(s.length-1)*q,i=Math.floor(p);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(p-i);}
  function solve(A,b){let n=b.length,M=A.map((r,i)=>[...r,b[i]]);for(let i=0;i<n;i++){let k=i;for(let j=i+1;j<n;j++)if(Math.abs(M[j][i])>Math.abs(M[k][i]))k=j;[M[i],M[k]]=[M[k],M[i]];if(Math.abs(M[i][i])<1e-12)throw Error('Singular fit. Remove redundant predictors or increase regularisation.');let d=M[i][i];for(let j=i;j<=n;j++)M[i][j]/=d;for(let k=0;k<n;k++)if(k!==i){let f=M[k][i];for(let j=i;j<=n;j++)M[k][j]-=f*M[i][j];}}return M.map(r=>r[n]);}
  function linear(X,y,w,penalty){let n=X[0].length,A=Array.from({length:n},()=>Array(n).fill(0)),b=Array(n).fill(0);for(let i=0;i<X.length;i++)for(let j=0;j<n;j++){b[j]+=w[i]*X[i][j]*y[i];for(let k=0;k<n;k++)A[j][k]+=w[i]*X[i][j]*X[i][k];}for(let j=1;j<n;j++)A[j][j]+=penalty;A[0][0]+=1e-10;return solve(A,b);}
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  let data,train,test,schema,models={},current=null;
  function init(d){data=d.rows;schema=d.schema;train=data.filter(r=>r.split==='train');test=data.filter(r=>r.split==='test');return {train:train.length,test:test.length};}
  const PAIRS=[['unresolved_support_requests','recent_price_change_pct'],...['business_size_band','business_sector','contract_plan','onboarding_route'].map(k=>['unresolved_support_requests',k]),...['business_size_band','business_sector'].map(k=>['recent_price_change_pct',k])];
  function prepare(keys,interactions=false){
    const groups=keys.map(key=>{
      const spec=schema.find(s=>s.column===key);if(!spec||!['core_feature','additional_feature','context'].includes(spec.role))throw Error('Not an eligible predictor: '+key);
      const vals=train.map(r=>r[key]).filter(v=>v!==null);
      if(spec.data_type==='number'){let med=quantile(vals,.5),full=train.map(r=>r[key]??med),mu=mean(full),sd=Math.sqrt(mean(full.map(v=>(v-mu)**2)))||1;return {key,type:'number',med,mu,sd,min:Math.min(...vals),max:Math.max(...vals),values:vals,unit:spec.unit};}
      return {key,type:'category',levels:[...new Set(vals)].sort(),unit:'category'};
    });
    const columns=[];groups.forEach((g,j)=>{g.indices=[];if(g.type==='number'){g.indices.push(columns.length);columns.push({group:j,label:g.key,scale:g.sd});}else g.levels.slice(1).forEach(l=>{g.indices.push(columns.length);columns.push({group:j,label:g.key+' = '+l,level:l,reference:g.levels[0],scale:1});});});
    const products=[];if(interactions)for(let pair of PAIRS){let a=groups.findIndex(g=>g.key===pair[0]),b=groups.findIndex(g=>g.key===pair[1]);if(a<0||b<0)continue;for(let i of groups[a].indices)for(let j of groups[b].indices){products.push([i,j]);columns.push({group:a,label:columns[i].label+' × '+columns[j].label,interaction:true,scale:1});}}
    function raw(r){return groups.map(g=>g.type==='number'?(r[g.key]??g.med):r[g.key]);}
    function encode(a){const out=[];groups.forEach((g,j)=>{if(g.type==='number')out.push((a[j]-g.mu)/g.sd);else g.levels.slice(1).forEach(l=>out.push(a[j]===l?1:0));});for(let [i,j] of products)out.push(out[i]*out[j]);return out;}
    return {groups,columns,raw,encode,X:train.map(r=>encode(raw(r))),Y:train.map(r=>r.cancels_within_30d)};
  }
  function logistic(prep,lambda){let X=prep.X.map(r=>[1,...r]),y=prep.Y,p=X[0].length,b=Array(p).fill(0),iterations=0,converged=false;
    for(let t=0;t<70;t++){let A=Array.from({length:p},()=>Array(p).fill(0)),grad=Array(p).fill(0);for(let i=0;i<X.length;i++){let prob=sigmoid(dot(X[i],b)),w=Math.max(1e-8,prob*(1-prob));for(let j=0;j<p;j++){grad[j]+=X[i][j]*(y[i]-prob);for(let k=0;k<p;k++)A[j][k]+=w*X[i][j]*X[i][k];}}for(let j=1;j<p;j++){A[j][j]+=lambda;grad[j]-=lambda*b[j];}A[0][0]+=1e-9;let step=solve(A,grad);b=b.map((v,j)=>v+step[j]);iterations=t+1;if(Math.max(...step.map(Math.abs))<1e-6){converged=true;break;}}
    return {kind:'logistic',coefficients:b,iterations,converged,predict:x=>sigmoid(dot([1,...x],b))};
  }
  function forest(prep,settings,progress){let r=rng(settings.seed),X=prep.X,y=prep.Y,p=X[0].length,trees=[];
    function build(ids,depth){let n=ids.length,s=ids.reduce((v,i)=>v+y[i],0),prob=s/n;
      if(depth>=settings.depth||n<2*settings.leaf||s===0||s===n)return {p:prob,n};
      let base=2*s*(n-s)/n,best=0,feature=-1,threshold=0;
      for(let f of shuffle(Array.from({length:p},(_,i)=>i),r).slice(0,Math.max(1,Math.floor(Math.sqrt(p))))){
        let sorted=ids.slice().sort((a,b)=>X[a][f]-X[b][f]),left=0;
        for(let k=0;k<n-1;k++){left+=y[sorted[k]];let a=k+1,b=n-a;if(a<settings.leaf||b<settings.leaf||X[sorted[k]][f]===X[sorted[k+1]][f])continue;let right=s-left,gain=base-2*left*(a-left)/a-2*right*(b-right)/b;if(gain>best+1e-10){best=gain;feature=f;threshold=(X[sorted[k]][f]+X[sorted[k+1]][f])/2;}}
      }
      if(feature<0)return {p:prob,n};let left=ids.filter(i=>X[i][feature]<=threshold),right=ids.filter(i=>X[i][feature]>threshold);
      return {f:feature,t:threshold,left:build(left,depth+1),right:build(right,depth+1),p:prob,n};
    }
    for(let t=0;t<settings.trees;t++){let ids=Array.from({length:X.length},()=>Math.floor(r()*X.length));trees.push(build(ids,0));if(t%10===0)progress?.('Fitting tree '+(t+1)+' of '+settings.trees);}
    function one(tree,x){while(tree.f!==undefined)tree=x[tree.f]<=tree.t?tree.left:tree.right;return tree.p;}
    return {kind:'forest',trees,predict:x=>mean(trees.map(t=>one(t,x)))};
  }
  function neural(prep,s,progress){
    const r=rng(s.seed),X=prep.X,Y=prep.Y,p=X[0].length,h=s.neurons,b1=p*h,w2=b1+h,b2=w2+h,n=b2+1;
    const theta=Array(n).fill(0),m=Array(n).fill(0),v=Array(n).fill(0),curve=[];
    for(let j=0;j<h;j++){for(let k=0;k<p;k++)theta[j*p+k]=(r()*2-1)*Math.sqrt(6/(p+h));theta[w2+j]=(r()*2-1)*Math.sqrt(6/(h+1));}
    function forward(x){let a=Array(h);for(let j=0;j<h;j++){let z=theta[b1+j];for(let k=0;k<p;k++)z+=theta[j*p+k]*x[k];a[j]=Math.tanh(z);}let z=theta[b2];for(let j=0;j<h;j++)z+=theta[w2+j]*a[j];return {a,prob:sigmoid(z)};}
    let step=0;
    for(let epoch=0;epoch<s.epochs;epoch++){
      let order=shuffle(Array.from({length:X.length},(_,i)=>i),r);
      for(let start=0;start<order.length;start+=64){let grad=Array(n).fill(0),batch=order.slice(start,start+64);
        for(let i of batch){let {a,prob}=forward(X[i]),d=prob-Y[i];grad[b2]+=d;for(let j=0;j<h;j++){grad[w2+j]+=d*a[j];let dh=d*theta[w2+j]*(1-a[j]*a[j]);grad[b1+j]+=dh;for(let k=0;k<p;k++)grad[j*p+k]+=dh*X[i][k];}}
        step++;for(let k=0;k<n;k++){let isWeight=k<b1||(k>=w2&&k<b2),g=grad[k]/batch.length+(isWeight?.001*theta[k]:0);g=clamp(g,-10,10);m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g;theta[k]-=s.learningRate*(m[k]/(1-Math.pow(.9,step)))/(Math.sqrt(v[k]/(1-Math.pow(.999,step)))+1e-8);}
      }
      if(epoch%10===0||epoch===s.epochs-1){let loss=mean(X.map((x,i)=>{let q=clamp(forward(x).prob,1e-9,1-1e-9);return -Y[i]*Math.log(q)-(1-Y[i])*Math.log(1-q);}));curve.push({epoch:epoch+1,loss});progress?.('Training neural network · epoch '+(epoch+1)+' of '+s.epochs);}
    }
    return {kind:'ann',weights:theta,trainingCurve:curve,architecture:[p,h,1],predict:x=>forward(x).prob};
  }
  function fit(config,progress){let settings={trees:80,depth:6,leaf:12,lambda:1,seed:5302,neurons:16,epochs:120,learningRate:.003,...config};
    if(!['logistic','forest','ann'].includes(settings.model))throw Error('Select a supported model.');if(!settings.features?.length)throw Error('Select at least one predictor.');
    for(let [k,lo,hi] of [['trees',10,200],['depth',1,10],['leaf',3,100],['lambda',.01,100],['neurons',4,32],['epochs',30,300],['learningRate',.0005,.02],['threshold',.01,.99],['seed',1,999999]]){if(!['threshold','seed',...(settings.model==='logistic'?['lambda']:settings.model==='forest'?['trees','depth','leaf']:['neurons','epochs','learningRate'])].includes(k))continue;let value=settings[k]??(k==='threshold'?.5:undefined);if(!Number.isFinite(value)||value<lo||value>hi)throw Error(k+' must be between '+lo+' and '+hi);if(['trees','depth','leaf','neurons','epochs','seed'].includes(k)&&!Number.isInteger(value))throw Error(k+' must be a whole number.');}
    let key=JSON.stringify(settings);if(models[key]){current=models[key];return current;}
    let prep=prepare(settings.features,settings.model==='logistic'&&settings.interactions!==false);progress?.('Preparing training data');let model=settings.model==='logistic'?logistic(prep,settings.lambda):settings.model==='ann'?neural(prep,settings,progress):forest(prep,settings,progress);
    model.prep=prep;model.settings=settings;model.predictRaw=x=>model.predict(prep.encode(x));model.rawTrain=train.map(prep.raw);model.rawTest=test.map(prep.raw);model.predictRow=r=>model.predictRaw(prep.raw(r));
    models[key]=model;while(Object.keys(models).length>4)delete models[Object.keys(models)[0]];current=model;return model;
  }
  function metrics(model){const probs=test.map(model.predictRow),y=test.map(r=>r.cancels_within_30d),threshold=model.settings.threshold??.5;let tp=0,tn=0,fp=0,fn=0;y.forEach((v,i)=>{if(probs[i]>=threshold)v?tp++:fp++;else v?fn++:tn++;});let pos=probs.filter((_,i)=>y[i]===1),neg=probs.filter((_,i)=>y[i]===0),wins=0;for(let a of pos)for(let b of neg)wins+=a>b?1:a===b?.5:0;
    return {threshold,tp,tn,fp,fn,n:y.length,trainN:train.length,positives:pos.length,accuracy:(tp+tn)/y.length,precision:tp+fp?tp/(tp+fp):null,recall:tp/(tp+fn),specificity:tn/(tn+fp),auc:wins/(pos.length*neg.length),brier:mean(y.map((v,i)=>(v-probs[i])**2)),baselineAccuracy:Math.max(pos.length,neg.length)/y.length,baselineBrier:mean(y.map(v=>(v-mean(model.prep.Y))**2))};
  }
  function sample(model,n=160){return shuffle(model.rawTrain,rng(901)).slice(0,n);}
  function grid(g,n=11){if(g.type==='category')return g.levels;let v=Array.from({length:n},(_,i)=>quantile(g.values,.05+.9*i/(n-1)));if(['count','days','months'].includes(g.unit))v=v.map(Math.round);return [...new Set(v)].sort((a,b)=>a-b);}
  function curve(model,key,kind){let j=model.prep.groups.findIndex(g=>g.key===key);if(j<0)throw Error('Fit a model containing '+key+' first.');let g=model.prep.groups[j],xs=grid(g),base=sample(model),lines=base.map(a=>xs.map(v=>{let b=a.slice();b[j]=v;return model.predictRaw(b);}));
    return {type:kind,feature:key,unit:g.unit,category:g.type==='category',x:xs,y:xs.map((_,k)=>mean(lines.map(l=>l[k]))),ice:kind==='ice'?lines.slice(0,24):undefined,n:base.length,scope:'Training reference sample; original other inputs retained. Grid uses 5th–95th percentiles (all category levels). Predictions may involve unusual feature combinations.'};
  }
  function ale(model,key){let j=model.prep.groups.findIndex(g=>g.key===key);if(j<0)throw Error('Fit a model containing '+key+' first.');let g=model.prep.groups[j];if(g.type!=='number')throw Error('ALE currently supports numerical predictors. Select ICE for a categorical predictor.');
    let edges=[...new Set(Array.from({length:11},(_,i)=>quantile(g.values,i/10)))],bins=[];if(edges.length<2)throw Error('ALE needs variation in the selected predictor.');
    for(let k=0;k<edges.length-1;k++){let a=edges[k],b=edges[k+1],rows=model.rawTrain.filter(x=>x[j]>=a&&(k===edges.length-2?x[j]<=b:x[j]<b));let diffs=rows.map(x=>{let low=x.slice(),high=x.slice();low[j]=a;high[j]=b;return model.predictRaw(high)-model.predictRaw(low);});bins.push({low:a,high:b,n:rows.length,delta:diffs.length?mean(diffs):0});}
    let total=0;bins.forEach(b=>{b.lower=total;total+=b.delta;b.upper=total;b.mid=(b.lower+b.upper)/2;});let centre=bins.reduce((s,b)=>s+b.mid*b.n,0)/model.rawTrain.length;
    return {type:'ale',feature:key,unit:g.unit,n:model.rawTrain.length,bins:bins.map(b=>({...b,effect:b.mid-centre})),x:bins.map(b=>(b.low+b.high)/2),y:bins.map(b=>b.mid-centre),scope:'First-order numerical ALE: quantile bins, within-bin prediction differences, midpoint accumulation centred by bin counts. Units are probability-point differences; this is not a probability or causal effect.'};
  }
  function coefficients(model){if(model.kind!=='logistic')throw Error('Odds ratios and marginal effects require a logistic regression model.');let p=sample(model,320),pred=p.map(model.predictRaw),slope=mean(pred.map(v=>v*(1-v)));
    return model.prep.columns.map((c,k)=>{let g=model.prep.groups[c.group],b=model.coefficients[k+1],ame;
      if(c.interaction)ame=null;else if(g.type==='number')ame=mean(p.map(a=>{let x=a.slice(),z=a.slice(),h=.001*g.sd;x[c.group]+=h;z[c.group]-=h;return (model.predictRaw(x)-model.predictRaw(z))/(2*h);}));else ame=mean(p.map(a=>{let x=a.slice(),z=a.slice();x[c.group]=c.level;z[c.group]=c.reference;return model.predictRaw(x)-model.predictRaw(z);}));
      return {feature:c.interaction?c.label:g.key,label:c.label,reference:c.interaction?'Product of standardised numeric / reference-coded categorical inputs':c.reference||'Per 1 '+g.unit,oddsRatio:Math.exp(b/c.scale),oddsRatioSD:Math.exp(b),logOddsSD:b,ame,ameSD:ame===null?null:g.type==='number'?ame*g.sd:ame,scale:g.type==='number'?g.sd:1};});
  }
  function interactionOutput(model,req){let key=req.features[0],j=model.prep.groups.findIndex(g=>g.key===key),field=req.field,b=model.prep.groups.findIndex(g=>g.key===field);if(j<0)throw Error('Include the focal predictor before fitting.');let g=model.prep.groups[j];if(g.type!=='number')throw Error('Choose a numerical focal predictor.');let levels=[...new Set(train.map(r=>r[field]))].sort(),base=sample(model,160),xs=grid(g),groups=[];
    for(let level of levels){let subset=field==='price_change_group'?shuffle(train.filter(r=>r[field]===level),rng(901)).slice(0,160).map(model.prep.raw):base;if(!subset.length)continue;let ys=xs.map(v=>mean(subset.map(a=>{let x=a.slice();if(b>=0)x[b]=level;x[j]=v;let h=.001*g.sd,z=x.slice();z[j]+=h;let q=model.predictRaw(x),q2=model.predictRaw(z);if(req.type==='marginal')return (q2-q)/h; if(req.type==='odds'){z[j]=v+1;return Math.exp(Math.log(model.predictRaw(z)/(1-model.predictRaw(z)))-Math.log(q/(1-q)));}return q;})));groups.push({label:level,x:xs,y:ys});}
    return {type:'interaction',output:req.type,feature:key,field,groups,scope:'Fitted logistic interaction exploration. Probability and marginal-effect curves use a common training reference with the group set to each category; price bands use their observed training subsets. Odds ratios compare a one-unit increase at each plotted value. These are associations, not causal effects.'};
  }
  function account(model,id){let row=data.find(r=>r.account_id.toLowerCase()===id.toLowerCase());if(!row)throw Error('Unknown account. Use AC00001–AC03200; test accounts are AC02401–AC03200.');return {id:row.account_id,split:row.split,x:model.prep.raw(row),probability:model.predictRow(row),imputed:model.prep.groups.filter(g=>row[g.key]===null).map(g=>g.key)};}
  function shap(model,id,orders=64){model.shapCache??={};const cacheKey=id+':'+orders;if(model.shapCache[cacheKey])return model.shapCache[cacheKey];let a=account(model,id),background=sample(model,16),r=rng(841),p=a.x.length,phi=Array(p).fill(0),bases=[];
    for(let k=0;k<orders;k++){let x=background[k%background.length].slice(),old=model.predictRaw(x);bases.push(old);for(let j of shuffle(Array.from({length:p},(_,i)=>i),r)){x[j]=a.x[j];let next=model.predictRaw(x);phi[j]+=(next-old)/orders;old=next;}}
    let base=mean(bases);return model.shapCache[cacheKey]={type:'shap',...a,base,orders,background:background.length,contributions:model.prep.groups.map((g,j)=>({feature:g.key,value:a.x[j],effect:phi[j]})),residual:a.probability-base-phi.reduce((s,v)=>s+v,0),scope:'Sampled interventional permutation SHAP on the probability scale. Original feature groups stay together. 64 random orders and 16 shared training-background records; approximate attributions, not exact TreeSHAP. Correlated features can make replacement combinations unrealistic.'};
  }
  function aggregateShap(model,records,progress){let sums=Array(model.settings.features.length).fill(0),signed=sums.slice(),accounts=[];records.forEach((r,i)=>{if(i%4===0)progress?.('Explaining account '+(i+1)+' of '+records.length);let a=shap(model,r.account_id);a.contributions.forEach((c,j)=>{sums[j]+=Math.abs(c.effect)/records.length;signed[j]+=c.effect/records.length;});accounts.push({id:a.id,probability:a.probability,contributions:a.contributions,residual:a.residual});});return {rows:model.prep.groups.map((g,j)=>({feature:g.key,effect:sums[j],signed:signed[j]})),accounts,meanProbability:mean(accounts.map(a=>a.probability)),n:records.length};}
  function cohortShap(model,field,progress){if(!['price_change_group','business_size_band','business_sector','contract_plan','onboarding_route'].includes(field))throw Error('Choose a business group.');let levels=[...new Set(test.map(r=>r[field]))].sort(),groups=[];
    for(let level of levels){let population=test.filter(r=>r[field]===level),records=shuffle(population,rng(109)).slice(0,32);progress?.('SHAP cohort: '+level);groups.push({label:level,populationN:population.length,suppressed:population.length<30,...(population.length>=30?aggregateShap(model,records,progress):{})});}
    return {type:'grouped_shap',field,groups,scope:'Each cohort uses up to 32 seeded holdout accounts, 64 permutations per account and the same 16 training backgrounds. Bars show mean absolute SHAP; signed means and individual accounts are also available. Groups below 30 holdout records are suppressed. Unequal group samples are not pooled into a global estimate. These are business-account comparisons, not causal effects or isolated SHAP interaction values.'};
  }
  function importance(model){let rows=model.rawTest.slice(),y=test.map(r=>r.cancels_within_30d),base=mean(rows.map((x,i)=>(model.predictRaw(x)-y[i])**2)),r=rng(917);return model.prep.groups.map((g,j)=>{let effects=[];for(let rep=0;rep<3;rep++){let perm=shuffle(rows.map(x=>x[j]),r);effects.push(mean(rows.map((x,i)=>{let z=x.slice();z[j]=perm[i];return (model.predictRaw(z)-y[i])**2;}))-base);}return {feature:g.key,effect:mean(effects)};});}
  function pairImportance(model){let base=metrics(model).brier,rows=model.rawTest,y=test.map(r=>r.cancels_within_30d);return PAIRS.filter(pair=>pair.every(k=>model.settings.features.includes(k))).map(pair=>{let idx=pair.map(k=>model.settings.features.indexOf(k)),effects=[];for(let rep=0;rep<3;rep++){let perm=shuffle(rows,rng(1700+rep));effects.push(mean(rows.map((x,i)=>{let z=x.slice();idx.forEach(j=>z[j]=perm[i][j]);return (model.predictRaw(z)-y[i])**2;}))-base);}return {feature:pair.join(' × '),effect:mean(effects)};});}
  function execute(cmd,progress){if(cmd.config)fit(cmd.config,progress);if(!current)throw Error('Run a model first.');let model=current,out=[];
    // Validate every requested output before calculating, so partial requests are not silently run.
    for(let req of cmd.outputs){let allowed=model.kind==='logistic'?['performance','odds','marginal','probabilities']:['performance','ice','ale','shap','global_shap','grouped_shap','importance'];if(!allowed.includes(req.type))throw Error('This explanation is not available for the selected model.');if(['ice','ale','probabilities'].includes(req.type)){let keys=req.features||model.settings.features;for(let k of keys){let g=model.prep.groups.find(g=>g.key===k);if(!g)throw Error('The fitted model does not include '+k+'.');if(req.type==='ale'&&g.type!=='number')throw Error('ALE supports numerical predictors only.');}}if(req.type==='shap')account(model,req.account);}
    for(let req of cmd.outputs){progress?.('Calculating '+req.type);switch(req.type){
      case 'performance':out.push({type:'performance',...metrics(model)});break;
      case 'odds':case 'marginal':if(req.field){out.push(interactionOutput(model,req));break;}out.push({type:req.type,rows:coefficients(model).filter(r=>req.type!=='marginal'||r.ame!==null),n:320,scope:'L2-regularised logistic regression. With interactions enabled, coefficient odds ratios describe reference-category / mean-numeric conditions; use grouped exploration for conditional odds ratios. Marginal effects use finite differences through all fitted terms. Average marginal effects are evaluated over 320 training-reference accounts; categorical changes compare with the stated reference. No inferential confidence intervals are supplied.'});break;
      case 'probabilities':case 'ice':if(req.field){out.push(interactionOutput(model,req));break;}for(let k of req.features||model.settings.features)out.push(curve(model,k,req.type));break;
      case 'ale':for(let k of req.features||model.prep.groups.filter(g=>g.type==='number').map(g=>g.key))out.push(ale(model,k));break;
      case 'shap':out.push(shap(model,req.account));break;
      case 'importance':out.push({type:'importance',rows:importance(model),pairs:pairImportance(model),scope:'Increase in holdout Brier error after permuting one original feature (3 repeats). Negative values are retained. Inspecting this holdout repeatedly can guide tuning and compromise its independence.'});break;
      case 'global_shap':out.push({type:'global_shap',...aggregateShap(model,shuffle(test,rng(109)).slice(0,80),progress),populationN:test.length,scope:'Global explanation estimated from 80 seeded holdout accounts out of 800. Mean absolute permutation SHAP, 64 orders and 16 training backgrounds. This is a sample estimate, not full-population importance.'});break;
      case 'grouped_shap':out.push(cohortShap(model,req.field,progress));break;
      default:throw Error('Unknown output.');
    }}
    return {model:model.kind,settings:model.settings,preprocessing:model.prep.groups.map(g=>({feature:g.key,type:g.type,median:g.med,mean:g.mu,sd:g.sd,reference:g.levels?.[0]})),converged:model.converged,iterations:model.iterations,outputs:out};
  }
  return {init,execute,fit,metrics,curve,ale,coefficients,shap,account,prepare,rng,solve,linear,sigmoid,quantile,getCurrent:()=>current};
}
if(typeof module!=='undefined')module.exports=createEngine;
