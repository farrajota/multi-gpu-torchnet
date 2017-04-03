--[[
    Compute mean/std statistics for the dataset.
]]


function datagen()
    print('Preparing train val and meanstd cache.')

    require 'image'
    local ffi = require 'ffi'
    local tnt = require 'torchnet'
    local t = require 'transforms.lua'
    local fix = t.Fix() -- Fixes number of channels
    local opt = opt
    local cache_path = paths.concat(opt.expDir, 'cache.t7')

    -- load dataset
    local dbc = require 'dbcollection.manager'
    local dbdataset = dbc.load{name = opt.dataset, data_dir=opt.data_dir}

    local result = {
       nClasses =  dbdataset:size('train', 'classes')[1],
       trainSize = dbdataset:size('train')[1]
    }

    -- get data loader function
    local data_loader = require 'data'
    local GetDataFn = data_loader(opt.dataset, 'train')

    local nSamples = math.min(result.trainSize, 10000)

    -----------------------------------------------------------
    -- Compute mean
    -----------------------------------------------------------

    print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')

    -- setup dataset iterator
    local iter = tnt.ListDataset{  -- replace this by your own dataset
       list = torch.range(1, dbdataset:size('train')[1]):long(),
       load = function(idx)
          local data = GetDataFn()
          return {
             input  = fix(data[1]):float(),
             target = data[2],
          }
       end
    }:shuffle(nSamples,true):iterator()

    local tm = torch.Timer()
    local meanEstimate = {0,0,0}
    local idx = 1
    xlua.progress(0, nSamples)
    for data in iter() do
       local img = data.input
       for j=1,3 do
          meanEstimate[j] = meanEstimate[j] + img[j]:mean()
       end
       idx = idx + 1
       if idx%100==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1,3 do
       meanEstimate[j] = meanEstimate[j] / nSamples
    end
    local mean = meanEstimate


    -----------------------------------------------------------
    -- Compute std
    -----------------------------------------------------------

    print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
    local stdEstimate = {0,0,0}
    idx = 1
    xlua.progress(0, nSamples)
    for data in iter() do
       local img = data.input
       for j=1,3 do
          stdEstimate[j] = stdEstimate[j] + img[j]:std()
       end
       idx = idx + 1
       if idx%100==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1,3 do
       stdEstimate[j] = stdEstimate[j] / nSamples
    end
    local std = stdEstimate


    -----------------------------------------------------------
    -- Store to disk
    -----------------------------------------------------------

    local cache = {}
    cache.mean = mean
    cache.std = std

    result.meanstd = cache
    print(cache)
    print('Time to estimate:', tm:time().real)
    torch.save(cache_path, result)
end

return datagen
