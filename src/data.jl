using JSON
function data(opts,splitname)
    fname = string(opts[:datapath],"/",splitname,"_processed.json")
    #sname = string(opts[:datapath],"/CLEVR_",splitname,"_scenes.json")
    sname = string(opts[:scenepath],"/CLEVR_",splitname,"_scenes.json")
    vfile = string(opts[:datapath],"/vocabulary.json")
    afile = string(opts[:datapath],"/answer_vocabulary.json")
    bs    = opts[:batchsize]
    # create dictionary from scenes data
    function scenedata()
        id2ix = Dict() # image id2 index
        mats =  Dict() # materials
        shas =  Dict() # shapes
        cols =  Dict() # colours
        sizs =  Dict() # sizes
        features = Dict()
        s = JSON.parsefile(sname)["scenes"]
        maxobjnum = 0
        for scene in s
            for obj in scene["objects"]
                get!(mats,obj["material"],length(mats)+1)
                get!(shas,obj["shape"],length(shas)+1)
                get!(cols,obj["color"],length(cols)+1)
                get!(sizs,obj["size"],length(sizs)+1)
            end
            maxobjnum = max(length(scene["objects"]), maxobjnum)
        end
        counter= 1
        for d in [mats,shas,cols,sizs]
            for m in keys(d)
                features[m]=counter
                counter +=1
            end
        end
        objlen = length(features)+3
        opts[:objlen] = objlen
        info("objects are represented with vectors length of $objlen")
        info("features:")
        display(features)
        info("----")
        # Order in objvector: material,shape,size,color,3dcoords
        scenematrix = zeros(Float32,maxobjnum*objlen,length(s))
        for (sindex,scene) in enumerate(s)
            id2ix[scene["image_index"]] = sindex
            for (oindex,obj) in enumerate(scene["objects"])
                scenematrix[(oindex-1)*objlen + get(features,obj["material"],-1),sindex] = 1
                scenematrix[(oindex-1)*objlen + get(features,obj["shape"],-1),sindex] = 1
                scenematrix[(oindex-1)*objlen + get(features,obj["size"],-1),sindex] = 1
                scenematrix[(oindex-1)*objlen + get(features,obj["color"],-1),sindex] = 1
                scenematrix[(oindex-1)*objlen + objlen-2:(oindex-1)*objlen + objlen,sindex] = obj["3d_coords"]
            end
        end
        info("scene matrices has been created...")
        return scenematrix,features,id2ix
    end # end inner function
    scenematrix,features,id2ix = scenedata()
    # process questions
    f=JSON.parsefile(fname)
    v = JSON.parsefile(vfile)["w2i"]
    info("vocabulary size:",length(v))
    opts[:vs] = length(v)
    a2i = JSON.parsefile(afile)
    info("read dataset & vocabulary")
    new_f = Any[]
    for i in f
        newd = i
        newd["encoded_question"] = Any[]
        newd["encoded_answer"] = Any[]
        for w in newd["final_question"]
            push!(newd["encoded_question"],v[w])
        end
        push!(newd["encoded_answer"],a2i[newd["answer"]])
        push!(new_f,newd)
    end
    info("encode questions")
    sorted_new_f = sort(new_f,lt=(x,y)->length(x["encoded_question"])<length(y["encoded_question"]),rev=true);
    result = Any[]
    imgix  = Any[]
    qix    = Any[]
    labels = Any[]
    for i=1:div(length(sorted_new_f),bs)
        batch = sorted_new_f[(i-1)*bs+1:i*bs]
        imgixbatch  = Any[]
        labelbatch = Any[]
        qixbatch = Any[]
        max_batch_len = maximum(map(x->length(x["encoded_question"]),batch))
        for j in batch
            push!(labelbatch,j["encoded_answer"][1])
            push!(imgixbatch,id2ix[j["image_index"]])
            push!(qixbatch,j["question_index"])
        end
        push!(result,batch)
        push!(imgix,imgixbatch)
        push!(labels,labelbatch)
        push!(qix,qixbatch)
    end
    final_result = Any[]
    iix = Any[]
    final_batchsizes = Any[]
    for j=1:length(result)
        batch = result[j]
        batchSizes = Int[]
        c = Int[]
        maxlen = maximum(map(x->length(x["encoded_question"]),batch))
        for i=1:maxlen
            bs_index = 0
            for k=1:length(batch)
                if length(batch[k]["encoded_question"]) >= i
                    bs_index += 1
                    push!(c,batch[k]["encoded_question"][i])
                end
            end
            push!(batchSizes,bs_index)
        end
        push!(final_result,c)
        push!(final_batchsizes,batchSizes)
        push!(iix,map(x->x["image_index"],batch))
    end
    # shuffle buckets
    info("shuffle data")
    indices = randperm(length(final_result))
    final_result = final_result[indices]
    final_batchsizes = final_batchsizes[indices]
    labels = labels[indices]
    imgix = imgix[indices]
    qix = qix[indices]
    info("textual data is ready")
    return final_result,final_batchsizes,labels,scenematrix,imgix,(features,id2ix,qix,iix)
end


#resdev,maskdev,qindev,iindev,smatdev,imgixdev,featdev,id2ixdev  = @time generatedata("../data/CLEVR_v1.0/questions/processed/dev_processed.json","../data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json","../data/CLEVR_v1.0/questions/processed/vocabulary.json",32);
#restrn,masktrn,qintrn,iintrn,smattrn,imgixtrn,feattrn,id2ixtrn = @time generatedata("../data/CLEVR_v1.0/questions/processed/train_processed.json","../data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json","../data/CLEVR_v1.0/questions/processed/vocabulary.json",32);
