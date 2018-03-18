using Knet
using Logging
using ArgParse
using JLD
include("data.jl")

import JLD: writeas, readas
import Knet: RNN
type RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
writeas(r::RNN) = RNNJLD(r.inputSize, r.hiddenSize, r.numLayers, r.dropout, r.inputMode, r.direction, r.mode, r.algo, r.dataType)
readas(r::RNNJLD) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout, skipInput=(r.inputMode==1), bidirectional=(r.direction==1), rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
type KnetJLD; a::Array; end
writeas(c::KnetArray) = KnetJLD(Array(c))
readas(d::KnetJLD) = (gpu() >= 0 ? KnetArray(d.a) : d.a)

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datapath";arg_type=String ; help="parsed data if available")
        ("--scenepath";arg_type=String ; help="scene matrices json file")
        ("--bestmodel";help="Save best model to file")
        ("--logfile";arg_type=String; help="log file")
        ("--epoch";arg_type=Int; default=1; help="Number of epochs for training.")
        ("--hidden";arg_type=Int;default=256)
        ("--embed";arg_type=Int;default=32)
        ("--batchsize";arg_type=Int; default=100)
        ("--seed";arg_type=Int; help="Random number seed.")
        ("--psize";arg_type=Int;default=28;help="Used in init weights (output size?).")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--pdrop";arg_type = Float64;nargs='+';default= [0.0,0.0,0.0];help="Dropout")
        ("--gclip";arg_type=Float64;default=5.0)
        ("--lr";arg_type=Float64;default=1e-3)
        ("--decayrate";arg_type=Float64;default=1.0)
        ("--optim";arg_type=String;help="Sgd|Adam")
        ("--patiance";arg_type=Int;default=10;help="number of validation to wait")
        ("--validate_per";arg_type=Int;default=10000;help="validate after that train sentences")
        ("--gs";nargs='+';default=[512,512,512,512];arg_type=Int;help="g-network hiddensize")
        ("--fs";nargs='+';default=[512,1024,29];arg_type=Int;help="f-network hiddensize")
    end
    return parse_args(s;as_symbols=true)
end

function main()
    opts = parseargs()
    setseed(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))
    opts[:model] = "relnet.jl"
    Logging.configure(level=DEBUG)
    Logging.configure(output=open(opts[:logfile], "a"))
    for (u,v) in opts
        info(u,"=>",v)
    end
    qtrn,bstrn,ytrn,smtrn,iixtrn,infotrn = data(opts,"train")
    qdev,bsdev,ydev,smdev,iixdev,infodev  = data(opts,"val")
    if isfile(opts[:bestmodel])
        # To continue training from the model given by opts[:bestmodel]
        info(" loading model and continue training...")
        mfile = JLD.load(opts[:bestmodel])
        weights = mfile["weights"]
        rsettings = mfile["rsettings"]
        opts[:lr] = mfile["lr"]
        params = mfile["params"]
        besties = zeros(2)
        besties[1] = mfile["bestacc"]
        besties[2] = 80.00
        optimtype = eval(parse(opts[:optim]))
        info("starting lr:",opts[:lr]," best devacc:",besties[1])

    else
        info("training from scratch...")
        rsettings,weights = initweights(opts)
        optimtype = eval(parse(opts[:optim]))
        params = oparams(weights,optimtype;lr=opts[:lr],gclip=opts[:gclip])
    end
    # global variables
    trn_max_objnum = div(size(smtrn,1),opts[:objlen])
    t1 = hcat(vec(transpose(repmat(1:trn_max_objnum,1,trn_max_objnum))),repmat(1:trn_max_objnum,trn_max_objnum,1))
    inds =  vec(transpose(t1))
    trn_pair_indices = vcat([ map(x->(i-1)*trn_max_objnum+x,inds) for i=1:opts[:batchsize]]...)
    trn_onesarr = opts[:atype](ones(Float32,1,trn_max_objnum * trn_max_objnum))

    dev_max_objnum = div(size(smdev,1),opts[:objlen])
    t1 = hcat(vec(transpose(repmat(1:dev_max_objnum,1,dev_max_objnum))),repmat(1:dev_max_objnum,dev_max_objnum,1))
    inds =  vec(transpose(t1))
    dev_pair_indices = vcat([ map(x->(i-1)*dev_max_objnum+x,inds) for i=1:opts[:batchsize]]...)
    dev_onesarr = opts[:atype](ones(Float32,1,dev_max_objnum * dev_max_objnum))

    besties  = zeros(2)
    patiance = zeros(1)
    patiance[1] = opts[:patiance]
    for i=1:opts[:epoch]
        lss = train(weights,rsettings,params,trn_onesarr,trn_pair_indices,qtrn,bstrn,ytrn,smtrn,iixtrn,opts,besties,patiance)
        info(@sprintf "epoch:%d trnlss:%.4f:" i lss[1]/lss[2])
        devlss,devacc = accuracy(weights,rsettings,dev_onesarr,dev_pair_indices,qdev,bsdev,ydev,smdev,iixdev,opts)
        info(@sprintf "[dev-%d] lss:%.4f acc:%.4f" i devlss[1]/devlss[2] devacc)
        if devacc > besties[1]
            besties[1] = devacc
            info("best dev accuracy: ",besties[1])
            JLD.save(opts[:bestmodel],"weights",weights,"rsettings",
                     rsettings,"params",params,"bestacc",besties[1],
                     "startfrom",i,"lr",opts[:lr])
            patiance[1] = opts[:patiance]
        else
            patiance[1] =  patiance[1] - 1
            if patiance[1] < 0
                info(@sprintf "Patiance goes below zero, training finalized, best dev acc:%.3f" besties[1])
                break
            end
            if patiance[1] == div(opts[:patiance],2)
                opts[:lr] = opts[:lr]*opts[:decayrate]
                params = oparams(weights,optimtype;lr=opts[:lr],gclip=opts[:gclip])
                info(@sprintf "learning rate has been set to: %.4f" opts[:lr])
            end
        end
    end
end

linear(w,x)=return w[1]*x.+w[2]
function relnet(weights,q_embedding,world,pair_indices,onesarr,pdrop)
    hs,bs  = size(q_embedding)
    qenc   = reshape(q_embedding,hs*bs,1)  # question encoding
    tqenc  = qenc * onesarr              # tailed question encoding
    tqenc =  reshape(permutedims(reshape(tqenc,hs,bs,size(onesarr,2)),[1,3,2]),hs,bs*size(onesarr,2))
    # create object pairs and merge question with object pairs
    objpairs = world[:,pair_indices]
    objpairs = reshape(objpairs,size(objpairs,1)*2,div(size(objpairs,2),2))
    # merged: 292x6400
    merged   = vcat(objpairs,tqenc)  # merged state description matrix & question encoding
    # G network
    outg1   = linear(weights[:G1],merged)
    outg1   = relu.(outg1)
    outg1   = dropout(outg1,pdrop[1])
    outg2   = linear(weights[:G2],outg1)
    outg2   = relu.(outg2)
    outg2   = dropout(outg2,pdrop[1])
    outg3   = linear(weights[:G3],outg2)
    outg3   = relu.(outg3)
    outg3   = dropout(outg3,pdrop[1])
    outg4   = linear(weights[:G4],outg3)
    outg4   = relu.(outg4)
    outg4   = dropout(outg4,pdrop[1])
    # sum object pairs
    spairs1 = reshape(outg4,size(outg4,1),size(onesarr,2),bs)
    spairs2 = reshape(sum(spairs1,2),size(spairs1,1),size(spairs1,3))
    # F network
    outf1   = linear(weights[:F1],spairs2)
    outf1   = relu.(outf1)
    outf1   = dropout(outf1,pdrop[2])
    outf2   = linear(weights[:F2],outf1)
    outf2   = relu.(outf2)
    outf2   = dropout(outf2,pdrop[2])
    outf3   = linear(weights[:F3],outf2)
    outf3   = relu.(outf3)
    outf3   = dropout(outf3,pdrop[2])
    return outf3
end

function logprob(output,ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

function loss(weights,rsettings,questions,bsizes,labels,world,pair_indices,onesarr;lss=nothing,pdrop=[0.0,0.0,0.0],preds=nothing)
    # question encoding
    embedding  = weights[:embedding][:,questions]
    ye,he,ce,_ = rnnforw(rsettings[1],weights[:lstm],embedding;hy=true,cy=true,batchSizes=bsizes)
    # relation module
    s1,s2,_ = size(he)
    rout = relnet(weights,reshape(he,s1,s2),world,pair_indices,onesarr,pdrop)
    scores = weights[:softmax][1] * rout .+ weights[:softmax][2]
    total = logprob(labels,scores)
    if lss != nothing
        lss[1] = lss[1] + AutoGrad.getval(-total)
        lss[2] = lss[2] + AutoGrad.getval(length(labels))
    end
    if preds != nothing
        push!(preds,AutoGrad.getval(scores))
    end
    return -total/length(labels)
end

lossgradient = grad(loss)

function train(weights,rsettings,params,onesarr,pair_indices,questions,batchsizes,labels,smatrix,imgix,opts,besties,patiance)
    lss = zeros(2)
    maxobjnum = div(size(smatrix,1),opts[:objlen])
    for i = 1:length(questions)
        world    = opts[:atype](smatrix[:,imgix[i]])
        world    = reshape(world,opts[:objlen],maxobjnum*opts[:batchsize])
        grads = lossgradient(weights,rsettings,questions[i],batchsizes[i],labels[i],world,pair_indices,onesarr;
                             lss=lss,pdrop=opts[:pdrop])
        update!(weights,grads,params)
    end
    return lss
end

function accuracy(weights,rsettings,onesarr,pair_indices,questions,batchsizes,labels,smatrix,imgix,opts)
    lss = zeros(2)
    cumsacc = Any[]
    maxobjnum = div(size(smatrix,1),opts[:objlen])
    for i = 1:length(questions)
        preds = Any[]
        world = opts[:atype](smatrix[:,imgix[i]])
        world = reshape(world,opts[:objlen],maxobjnum*opts[:batchsize])
        loss(weights,rsettings,questions[i],batchsizes[i],labels[i],world,pair_indices,onesarr;lss=lss,preds=preds)
        acc = (mapslices(indmax,Array(preds[1]),1) .== reshape(labels[i],1,opts[:batchsize]))
        push!(cumsacc,acc...)
    end
    return lss,sum(cumsacc)/length(cumsacc)
end

function initweights(opts)
    weights  = Dict()
    rsettings = Any[]
    gsize    = opts[:gs]
    psize    = opts[:psize]
    fsize    = opts[:fs]
    objlen   = opts[:objlen]
    atype    = opts[:atype]
    init     = xavier
    # Embedding
    weights[:embedding] = atype(init(opts[:embed],opts[:vs]))
    # Question encoder
    r1,lstm = rnninit(opts[:embed],opts[:hidden];rnnType=:lstm,binit=zeros)
    push!(rsettings,r1)
    weights[:lstm] = lstm
    # G Network
    weights[:G1] = [atype(init(gsize[1],2*objlen + opts[:hidden])),atype(zeros(gsize[1],1))]
    weights[:G2] = [atype(init(gsize[2],gsize[1])),atype(zeros(gsize[2],1))]
    weights[:G3] = [atype(init(gsize[3],gsize[2])),atype(zeros(gsize[3],1))]
    weights[:G4] = [atype(init(gsize[4],gsize[3])),atype(zeros(gsize[4],1))]
    # F Network
    weights[:F1] = [atype(init(fsize[1],gsize[4])),atype(zeros(fsize[1],1))]
    weights[:F2] = [atype(init(fsize[2],fsize[1])),atype(zeros(fsize[2],1))]
    weights[:F3] = [atype(init(fsize[3],fsize[2])),atype(zeros(fsize[3],1))]
    # Prediction
    weights[:softmax] = [atype(init(psize,fsize[3])),atype(init(psize,1))]
    return rsettings,weights
end

oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

!isinteractive() && main()
